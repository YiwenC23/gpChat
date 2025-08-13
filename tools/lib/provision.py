#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import logging
import os
import platform
import subprocess
import sys
from typing import NoReturn

os.environ["PYTHONUNBUFFERED"] = "y"

ZULIP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(ZULIP_PATH)

from scripts.lib.node_cache import setup_node_modules
from scripts.lib.setup_venv import get_venv_dependencies
from scripts.lib.zulip_tools import (
    ENDC,
    FAIL,
    WARNING,
    get_dev_uuid_var_path,
    os_families,
    parse_os_release,
    run,
    run_as_root,
)

VAR_DIR_PATH = os.path.join(ZULIP_PATH, "var")

CONTINUOUS_INTEGRATION = "GITHUB_ACTIONS" in os.environ

if not os.path.exists(os.path.join(ZULIP_PATH, ".git")):
    print(FAIL + "Error: No Zulip Git repository present!" + ENDC)
    print("To set up the Zulip development environment, you should clone the code")
    print("from GitHub, rather than using a Zulip production release tarball.")
    sys.exit(1)

# Check the RAM on the user's system, and throw an effort if <1.5GB.
# This avoids users getting segfaults running `pip install` that are
# generally more annoying to debug.
with open("/proc/meminfo") as meminfo:
    ram_size = meminfo.readlines()[0].strip().split(" ")[-2]
ram_gb = float(ram_size) / 1024.0 / 1024.0
if ram_gb < 1.5:
    print(
        f"You have insufficient RAM ({round(ram_gb, 2)} GB) to run the Zulip development environment."
    )
    print("We recommend at least 2 GB of RAM, and require at least 1.5 GB.")
    sys.exit(1)

try:
    UUID_VAR_PATH = get_dev_uuid_var_path(create_if_missing=True)
    os.makedirs(UUID_VAR_PATH, exist_ok=True)
    if os.path.exists(os.path.join(VAR_DIR_PATH, "zulip-test-symlink")):
        os.remove(os.path.join(VAR_DIR_PATH, "zulip-test-symlink"))
    os.symlink(
        os.path.join(ZULIP_PATH, "README.md"),
        os.path.join(VAR_DIR_PATH, "zulip-test-symlink"),
    )
    os.remove(os.path.join(VAR_DIR_PATH, "zulip-test-symlink"))
except OSError:
    print(
        FAIL + "Error: Unable to create symlinks. "
        "Make sure you have permission to create symbolic links." + ENDC
    )
    print("See this page for more information:")
    print(
        "  https://zulip.readthedocs.io/en/latest/development/setup-recommended.html#os-symlink-error"
    )
    sys.exit(1)

distro_info = parse_os_release()
vendor = distro_info["ID"]
os_version = distro_info["VERSION_ID"]
if vendor == "debian" and os_version == "12":  # bookworm
    POSTGRESQL_VERSION = "15"
elif vendor == "ubuntu" and os_version == "22.04":  # jammy
    POSTGRESQL_VERSION = "14"
elif vendor == "ubuntu" and os_version == "24.04":  # noble
    POSTGRESQL_VERSION = "16"
elif vendor == "fedora" and os_version == "38":
    POSTGRESQL_VERSION = "15"
elif vendor == "rhel" and os_version.startswith("7."):
    POSTGRESQL_VERSION = "10"
elif vendor == "centos" and os_version == "7":
    POSTGRESQL_VERSION = "10"
else:
    logging.critical("Unsupported platform: %s %s", vendor, os_version)
    sys.exit(1)

VENV_DEPENDENCIES = get_venv_dependencies(vendor, os_version)

COMMON_DEPENDENCIES = [
    "memcached",
    "rabbitmq-server",
    "supervisor",
    "git",
    "curl",
    "ca-certificates",  # Explicit dependency in case e.g. curl is already installed
    "puppet",  # Used by lint (`puppet parser validate`)
    "gettext",  # Used by makemessages i18n
    "curl",  # Used for testing our API documentation
    "moreutils",  # Used for sponge command
    "unzip",  # Needed for Slack import
    "crudini",  # Used for shell tooling w/ zulip.conf
    # Puppeteer dependencies from here
    "xdg-utils",
    # Puppeteer dependencies end here.
]

UBUNTU_COMMON_APT_DEPENDENCIES = [
    *COMMON_DEPENDENCIES,
    "redis-server",
    "hunspell-en-us",
    "puppet-lint",
    "default-jre-headless",  # Required by vnu-jar
    # Puppeteer dependencies from here
    "fonts-freefont-ttf",
    "libatk-bridge2.0-0",
    "libgbm1",
    "libgtk-3-0",
    "libx11-xcb1",
    "libxcb-dri3-0",
    "libxss1",
    "xvfb",
    # Puppeteer dependencies end here.
]

COMMON_YUM_DEPENDENCIES = [
    *COMMON_DEPENDENCIES,
    "redis",
    "hunspell-en-US",
    "rubygem-puppet-lint",
    "nmap-ncat",
    "ccache",  # Required to build pgroonga from source.
    # Puppeteer dependencies from here
    "at-spi2-atk",
    "GConf2",
    "gtk3",
    "libX11-xcb",
    "libxcb",
    "libXScrnSaver",
    "mesa-libgbm",
    "xorg-x11-server-Xvfb",
    # Puppeteer dependencies end here.
]

BUILD_GROONGA_FROM_SOURCE = False
BUILD_PGROONGA_FROM_SOURCE = False
BUILD_PGVECTORSCALE_FROM_SOURCE = False
ENABLE_PGVECTORSCALE = False

# Check if PostgreSQL version supports pgvector/pgvectorscale (>= 13)
PG_MAJOR = int(POSTGRESQL_VERSION.split(".")[0]) if "." in POSTGRESQL_VERSION else int(POSTGRESQL_VERSION)
if PG_MAJOR >= 13:
    # Enable pgvectorscale on supported platforms
    if vendor in ["debian", "ubuntu"]:
        ENABLE_PGVECTORSCALE = True
        BUILD_PGVECTORSCALE_FROM_SOURCE = True
    elif vendor == "fedora":
        ENABLE_PGVECTORSCALE = True
        BUILD_PGVECTORSCALE_FROM_SOURCE = True

if (vendor == "debian" and os_version in []) or (vendor == "ubuntu" and os_version in []):
    # For platforms without a PGroonga release, we need to build it
    # from source.
    BUILD_PGROONGA_FROM_SOURCE = True
    SYSTEM_DEPENDENCIES = [
        *UBUNTU_COMMON_APT_DEPENDENCIES,
        f"postgresql-{POSTGRESQL_VERSION}",
        # Dependency for building PGroonga from source
        f"postgresql-server-dev-{POSTGRESQL_VERSION}",
        "libgroonga-dev",
        "libmsgpack-dev",
        "clang",
        *VENV_DEPENDENCIES,
    ]
elif "debian" in os_families():
    DEBIAN_DEPENDENCIES = UBUNTU_COMMON_APT_DEPENDENCIES

    # If we are on an aarch64 processor, ninja will be built from source,
    # so cmake is required
    if platform.machine() == "aarch64":
        DEBIAN_DEPENDENCIES.append("cmake")

    BASE_DEPENDENCIES = [
        *DEBIAN_DEPENDENCIES,
        f"postgresql-{POSTGRESQL_VERSION}",
        f"postgresql-{POSTGRESQL_VERSION}-pgroonga",
        *VENV_DEPENDENCIES,
    ]

    # Add pgvector and build dependencies if PostgreSQL >= 13
    if ENABLE_PGVECTORSCALE:
        BASE_DEPENDENCIES.extend([
            f"postgresql-{POSTGRESQL_VERSION}-pgvector",  # pgvector from apt
            f"postgresql-server-dev-{POSTGRESQL_VERSION}",  # for building extensions
            "build-essential",
            "clang",
            "llvm",
            "pkg-config",
            "libssl-dev",
            "jq",  # for parsing cargo metadata
        ])

    SYSTEM_DEPENDENCIES = BASE_DEPENDENCIES
elif "rhel" in os_families():
    SYSTEM_DEPENDENCIES = [
        *COMMON_YUM_DEPENDENCIES,
        f"postgresql{POSTGRESQL_VERSION}-server",
        f"postgresql{POSTGRESQL_VERSION}",
        f"postgresql{POSTGRESQL_VERSION}-devel",
        f"postgresql{POSTGRESQL_VERSION}-pgdg-pgroonga",
        *VENV_DEPENDENCIES,
    ]
elif "fedora" in os_families():
    SYSTEM_DEPENDENCIES = [
        *COMMON_YUM_DEPENDENCIES,
        f"postgresql{POSTGRESQL_VERSION}-server",
        f"postgresql{POSTGRESQL_VERSION}",
        f"postgresql{POSTGRESQL_VERSION}-devel",
        # Needed to build PGroonga from source
        "msgpack-devel",
        *VENV_DEPENDENCIES,
    ]
    BUILD_GROONGA_FROM_SOURCE = True
    BUILD_PGROONGA_FROM_SOURCE = True

if "fedora" in os_families():
    TSEARCH_STOPWORDS_PATH = f"/usr/pgsql-{POSTGRESQL_VERSION}/share/tsearch_data/"
else:
    TSEARCH_STOPWORDS_PATH = f"/usr/share/postgresql/{POSTGRESQL_VERSION}/tsearch_data/"
REPO_STOPWORDS_PATH = os.path.join(
    ZULIP_PATH,
    "puppet",
    "zulip",
    "files",
    "postgresql",
    "zulip_english.stop",
)


def install_ollama() -> None:
    """Install Ollama AI runtime and download base models"""
    print(f"\n=====Installing Ollama AI runtime=====")

    import time, shutil, textwrap, tempfile

    # Check if Ollama is already installed
    ollama_installed = False
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            ollama_installed = True
            print("Ollama already installed, checking version...")
            run(["ollama", "--version"])
    except Exception:
        pass

    if not ollama_installed:
        print("Installing Ollama...")
        # Install Ollama using official script with proper error handling
        try:
            # For Ubuntu/Debian systems
            if os.path.exists("/etc/debian_version"):
                tmp_tgz = "/tmp/ollama.tgz"
                # Check if is arm64 architecture or amd64
                if platform.machine() == "aarch64":
                    print("Installing Ollama for arm64 architecture...")
                    # Download and install using the official method for arm64
                    run_as_root(["curl", "-L", "https://ollama.com/download/ollama-linux-arm64.tgz", "-o", f"{tmp_tgz}"])
                elif platform.machine() == "x86_64":
                    print("Installing Ollama for x86_64 architecture...")
                    # Download and install using the official method for amd64
                    run_as_root(["curl", "-L", "https://ollama.com/download/ollama-linux-amd64.tgz", "-o", f"{tmp_tgz}"])
                run_as_root(["tar", "-C", "/usr", "-xzf", f"{tmp_tgz}"])
                run_as_root(["rm", "-f", f"{tmp_tgz}"]) # Clean up the temporary file
            else:
                print("Warning: Automated Ollama installation only supported on Debian/Ubuntu")
                print("Please install Ollama manually: https://ollama.com/download")

            # Configure Ollama Service
            # Check if ollama user exists, if not create it
            try:
                subprocess.run(["id", "ollama"], capture_output=True, check=True)
                print("Ollama user already exists.")
            except subprocess.CalledProcessError:
                print("Creating ollama user...")
                run_as_root(["useradd", "-r", "-s", "/bin/false", "-U", "-m", "-d", "/usr/share/ollama", "ollama"])
                print("Adding current user to ollama group...")
                run_as_root(["bash", "-c", "usermod -a -G ollama $(whoami)"])

            service_path = "/etc/systemd/system/ollama.service"
            ollama_path = shutil.which("ollama")

            service_content = textwrap.dedent(f"""\
                [Unit]
                Description=Ollama Service
                After=network-online.target

                [Service]
                ExecStart={ollama_path} serve
                User=ollama
                Group=ollama
                Restart=always
                RestartSec=3
                Environment="PATH=$PATH"
                Environment="OLLAMA_HOST=0.0.0.0"
                Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"

                [Install]
                WantedBy=multi-user.target
            """)

            print("Creating Ollama service file...")
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(service_content)
                temp_path = tmp.name
            run_as_root(["cp", temp_path, service_path])
            print("Ollama service file created at", service_path)
            run_as_root(["rm", "-f", f"{temp_path}"]) # Clean up the temporary file

            print("Reload systemd and enable/start Ollama service...")
            try:
                run_as_root(["systemctl", "daemon-reload"])
                run_as_root(["systemctl", "enable", "ollama"])
                run_as_root(["systemctl", "start", "ollama"], sudo_args=["-H"])

                # Give the service a moment to start
                time.sleep(5)

                # Check if the service started successfully
                result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True)
                if result.returncode != 0:
                    print("Warning: Ollama service did not start successfully.")
                    print("You may need to start it manually with: ollama serve")
                else:
                    print("Ollama service started successfully.")
            except Exception as e:
                print(f"Warning: Could not start Ollama service: {e}")
                print("You may need to start it manually with: ollama serve")
                pass
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install Ollama: {e}")
            print("Please install Ollama manually: https://ollama.com/download")
            pass


def download_ollama_models():
    import shutil
    # Check disk space before downloading models
    def check_disk_space(path="/usr/share/ollama/.ollama/models", required_gb=10):
        """Check if there's enough disk space for models"""
        try:
            stat = shutil.disk_usage(path if os.path.exists(path) else "/")
            available_gb = stat.free / (1024**3)
            return available_gb >= required_gb, available_gb
        except Exception:
            return True, 0  # Assume enough space if check fails

    # Model sizes in GB (approximate)
    models_to_download = [
        ("llama3.1:8b", 4.9),       # Main language model ~4.9GB
        ("nomic-embed-text:v1.5", 0.3),  # Embedding model ~300MB
    ]

    # Check total required space
    total_required_gb = sum(size for _, size in models_to_download) + 1  # +1GB buffer
    has_space, available_gb = check_disk_space()

    if not has_space:
        print(f"Warning: Insufficient disk space for model downloads!")
        print(f"Required: ~{total_required_gb:.1f}GB, Available: {available_gb:.1f}GB")
        print("Skipping model downloads. You can download them later when you have more space:")
        for model, _ in models_to_download:
            print(f"  ollama pull {model}")
        print("Continuing with the rest of the provisioning process...")
        return  # Skip model downloads but continue provisioning

    # Download models with individual error handling
    for model, size_gb in models_to_download:
        try:
            # Check space before each download
            has_space, available_gb = check_disk_space()
            if available_gb < size_gb + 0.5:  # Need model size + 0.5GB buffer
                print(f"Warning: Insufficient space for {model} (needs ~{size_gb}GB, have {available_gb:.1f}GB)")
                print(f"Skipping {model}. You can download it later with: ollama pull {model}")
                continue

            print(f"* Downloading {model} (~{size_gb}GB)...")

            # Run with timeout to prevent hanging
            result = subprocess.run(
                ["ollama", "pull", model],
                timeout=1800,  # 30 minute timeout
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✓ Successfully downloaded {model}")
            else:
                # Check if it's a disk space error
                if "no space" in result.stderr.lower() or "disk full" in result.stderr.lower():
                    print(f"✗ Insufficient disk space while downloading {model}")
                    print(f"  Free up space and run: ollama pull {model}")
                else:
                    print(f"✗ Failed to download {model}: {result.stderr}")
                    print(f"  You can try again later with: ollama pull {model}")

        except subprocess.TimeoutExpired:
            print(f"✗ Download of {model} timed out after 30 minutes")
            print(f"  You can try again later with: ollama pull {model}")
            pass
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {model}: {e}")
            print(f"  You can try again later with: ollama pull {model}")
            pass
        except Exception as e:
            print(f"✗ Unexpected error downloading {model}: {e}")
            print(f"  You can try again later with: ollama pull {model}")
            pass

    print("Ollama models downloaded successfully!")


def install_system_deps() -> None:
    # By doing list -> set -> list conversion, we remove duplicates.
    deps_to_install = sorted(set(SYSTEM_DEPENDENCIES))

    if "fedora" in os_families():
        install_yum_deps(deps_to_install)
    elif "debian" in os_families():
        install_apt_deps(deps_to_install)
    else:
        raise AssertionError("Invalid vendor")

    # For some platforms, there aren't published PGroonga
    # packages available, so we build them from source.
    if BUILD_GROONGA_FROM_SOURCE:
        run_as_root(["./scripts/lib/build-groonga"])
    if BUILD_PGROONGA_FROM_SOURCE:
        run_as_root(["./scripts/lib/build-pgroonga"])

    # Build pgvectorscale if enabled and needed
    if BUILD_PGVECTORSCALE_FROM_SOURCE and ENABLE_PGVECTORSCALE:
        print("\n=====Building pgvectorscale extension=====")
        try:
            run_as_root(["./scripts/lib/build-pgvectorscale"])
            print("pgvectorscale build completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to build pgvectorscale: {e}")
            print("You can try building it manually later with: sudo ./scripts/lib/build-pgvectorscale")


def install_apt_deps(deps_to_install: list[str]) -> None:
    # setup-apt-repo does an `apt-get update` if the sources.list files changed.
    run_as_root(["./scripts/lib/setup-apt-repo"])

    # But we still need to do our own to make sure we have up-to-date
    # data before installing new packages, as the system might not have
    # done an apt update in weeks otherwise, which could result in 404s
    # trying to download old versions that were already removed from mirrors.
    run_as_root(["apt-get", "update"])
    run_as_root(
        [
            "env",
            "DEBIAN_FRONTEND=noninteractive",
            "apt-get",
            "-y",
            "install",
            "--allow-downgrades",
            "--no-install-recommends",
            *deps_to_install,
        ]
    )

    # Install Ollama AI runtime (always installed for AI agent support)
    install_ollama()
    download_ollama_models()


def install_yum_deps(deps_to_install: list[str]) -> None:
    print(WARNING + "RedHat support is still experimental." + ENDC)
    run_as_root(["./scripts/lib/setup-yum-repo"])

    # Hack specific to unregistered RHEL system.  The moreutils
    # package requires a perl module package, which isn't available in
    # the unregistered RHEL repositories.
    #
    # Error: Package: moreutils-0.49-2.el7.x86_64 (epel)
    #        Requires: perl(IPC::Run)
    yum_extra_flags: list[str] = []
    if vendor == "rhel":
        proc = subprocess.run(
            ["sudo", "subscription-manager", "status"],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 1:
            # TODO this might overkill since `subscription-manager` is already
            # called in setup-yum-repo
            if "Status" in proc.stdout:
                # The output is well-formed
                yum_extra_flags = ["--skip-broken"]
            else:
                print("Unrecognized output. `subscription-manager` might not be available")

    run_as_root(["yum", "install", "-y", *yum_extra_flags, *deps_to_install])
    if "rhel" in os_families():
        # This is how a pip3 is installed to /usr/bin in CentOS/RHEL
        # for python35 and later.
        run_as_root(["python36", "-m", "ensurepip"])
        # `python36` is not aliased to `python3` by default
        run_as_root(["ln", "-nsf", "/usr/bin/python36", "/usr/bin/python3"])
    postgresql_dir = f"pgsql-{POSTGRESQL_VERSION}"
    for cmd in ["pg_config", "pg_isready", "psql"]:
        # Our tooling expects these PostgreSQL scripts to be at
        # well-known paths.  There's an argument for eventually
        # making our tooling auto-detect, but this is simpler.
        run_as_root(["ln", "-nsf", f"/usr/{postgresql_dir}/bin/{cmd}", f"/usr/bin/{cmd}"])

    # From here, we do the first-time setup/initialization for the PostgreSQL database.
    pg_datadir = f"/var/lib/pgsql/{POSTGRESQL_VERSION}/data"
    pg_hba_conf = os.path.join(pg_datadir, "pg_hba.conf")

    # We can't just check if the file exists with os.path, since the
    # current user likely doesn't have permission to read the
    # pg_datadir directory.
    if subprocess.call(["sudo", "test", "-e", pg_hba_conf]) == 0:
        # Skip setup if it has been applied previously
        return

    run_as_root(
        [f"/usr/{postgresql_dir}/bin/postgresql-{POSTGRESQL_VERSION}-setup", "initdb"],
        sudo_args=["-H"],
    )
    # Use vendored pg_hba.conf, which enables password authentication.
    run_as_root(["cp", "-a", "puppet/zulip/files/postgresql/centos_pg_hba.conf", pg_hba_conf])
    # Later steps will ensure PostgreSQL is started

    # Link in tsearch data files
    if vendor == "fedora":
        # Since F36 dictionary files were moved away from /usr/share/myspell
        tsearch_source_prefix = "/usr/share/hunspell"
    else:
        tsearch_source_prefix = "/usr/share/myspell"
    run_as_root(
        [
            "ln",
            "-nsf",
            os.path.join(tsearch_source_prefix, "en_US.dic"),
            f"/usr/pgsql-{POSTGRESQL_VERSION}/share/tsearch_data/en_us.dict",
        ]
    )
    run_as_root(
        [
            "ln",
            "-nsf",
            os.path.join(tsearch_source_prefix, "en_US.aff"),
            f"/usr/pgsql-{POSTGRESQL_VERSION}/share/tsearch_data/en_us.affix",
        ]
    )


def main(options: argparse.Namespace) -> NoReturn:
    # pnpm and management commands expect to be run from the root of the
    # project.
    os.chdir(ZULIP_PATH)

    # hash the apt dependencies
    sha_sum = hashlib.sha1()

    for apt_dependency in SYSTEM_DEPENDENCIES:
        sha_sum.update(apt_dependency.encode())
    if "debian" in os_families():
        with open("scripts/lib/setup-apt-repo", "rb") as fb:
            sha_sum.update(fb.read())
    else:
        # hash the content of setup-yum-repo*
        with open("scripts/lib/setup-yum-repo", "rb") as fb:
            sha_sum.update(fb.read())

    # hash the content of build-pgroonga if Groonga is built from source
    if BUILD_GROONGA_FROM_SOURCE:
        with open("scripts/lib/build-groonga", "rb") as fb:
            sha_sum.update(fb.read())

    # hash the content of build-pgroonga if PGroonga is built from source
    if BUILD_PGROONGA_FROM_SOURCE:
        with open("scripts/lib/build-pgroonga", "rb") as fb:
            sha_sum.update(fb.read())

    # hash the content of build-pgvectorscale if pgvectorscale is built from source
    if BUILD_PGVECTORSCALE_FROM_SOURCE and ENABLE_PGVECTORSCALE:
        if os.path.exists("scripts/lib/build-pgvectorscale"):
            with open("scripts/lib/build-pgvectorscale", "rb") as fb:
                sha_sum.update(fb.read())

    new_apt_dependencies_hash = sha_sum.hexdigest()
    last_apt_dependencies_hash = None
    apt_hash_file_path = os.path.join(UUID_VAR_PATH, "apt_dependencies_hash")
    with open(apt_hash_file_path, "a+") as hash_file:
        hash_file.seek(0)
        last_apt_dependencies_hash = hash_file.read()

    if new_apt_dependencies_hash != last_apt_dependencies_hash:
        try:
            install_system_deps()
        except subprocess.CalledProcessError:
            try:
                # Might be a failure due to network connection issues. Retrying...
                print(WARNING + "Installing system dependencies failed; retrying..." + ENDC)
                install_system_deps()
            except BaseException as e:
                # Suppress exception chaining
                raise e from None
        with open(apt_hash_file_path, "w") as hash_file:
            hash_file.write(new_apt_dependencies_hash)
    else:
        print("No changes to apt dependencies, so skipping apt operations.")

    # Here we install node.
    proxy_env = [
        "env",
        "http_proxy=" + os.environ.get("http_proxy", ""),
        "https_proxy=" + os.environ.get("https_proxy", ""),
        "no_proxy=" + os.environ.get("no_proxy", ""),
    ]
    run_as_root([*proxy_env, "scripts/lib/install-node"], sudo_args=["-H"])

    try:
        setup_node_modules()
    except subprocess.CalledProcessError:
        print(WARNING + "`pnpm install` failed; retrying..." + ENDC)
        try:
            setup_node_modules()
        except subprocess.CalledProcessError:
            print(
                FAIL
                + "`pnpm install` is failing; check your network connection (and proxy settings)."
                + ENDC
            )
            sys.exit(1)

    # Install shellcheck.
    run_as_root([*proxy_env, "tools/setup/install-shellcheck"])
    # Install shfmt.
    run_as_root([*proxy_env, "tools/setup/install-shfmt"])

    # Install transifex-cli.
    run_as_root([*proxy_env, "tools/setup/install-transifex-cli"])

    # Install tusd
    run_as_root([*proxy_env, "tools/setup/install-tusd"])

    # Install Python environment
    run_as_root([*proxy_env, "scripts/lib/install-uv"])
    run(
        [*proxy_env, "uv", "sync", "--frozen"],
        env={k: v for k, v in os.environ.items() if k not in {"PYTHONDEVMODE", "PYTHONWARNINGS"}},
    )
    # Clean old symlinks used before uv migration
    with contextlib.suppress(FileNotFoundError):
        os.unlink("zulip-py3-venv")
    if os.path.lexists("/srv/zulip-py3-venv"):
        run_as_root(["rm", "/srv/zulip-py3-venv"])

    run_as_root(["cp", REPO_STOPWORDS_PATH, TSEARCH_STOPWORDS_PATH])

    if CONTINUOUS_INTEGRATION and not options.is_build_release_tarball_only:
        run_as_root(["service", "redis-server", "start"])
        run_as_root(["service", "memcached", "start"])
        run_as_root(["service", "rabbitmq-server", "start"])
        run_as_root(["service", "postgresql", "start"])
    elif "fedora" in os_families():
        # These platforms don't enable and start services on
        # installing their package, so we do that here.
        for service in [
            f"postgresql-{POSTGRESQL_VERSION}",
            "rabbitmq-server",
            "memcached",
            "redis",
        ]:
            run_as_root(["systemctl", "enable", service], sudo_args=["-H"])
            run_as_root(["systemctl", "start", service], sudo_args=["-H"])

    # If we imported modules after activating the virtualenv in this
    # Python process, they could end up mismatching with modules we’ve
    # already imported from outside the virtualenv.  That seems like a
    # bad idea, and empirically it can cause Python to segfault on
    # certain cffi-related imports.  Instead, start a new Python
    # process inside the virtualenv.
    provision_inner = os.path.join(ZULIP_PATH, "tools", "lib", "provision_inner.py")
    os.execvp(
        "uv",
        [
            "uv",
            "run",
            "--no-sync",
            provision_inner,
            *(["--force"] if options.is_force else []),
            *(["--build-release-tarball-only"] if options.is_build_release_tarball_only else []),
            *(["--skip-dev-db-build"] if options.skip_dev_db_build else []),
        ],
    )


if __name__ == "__main__":
    description = "Provision script to install Zulip"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--force",
        action="store_true",
        dest="is_force",
        help="Ignore all provisioning optimizations.",
    )

    parser.add_argument(
        "--build-release-tarball-only",
        action="store_true",
        dest="is_build_release_tarball_only",
        help="Provision needed to build release tarball.",
    )

    parser.add_argument(
        "--skip-dev-db-build", action="store_true", help="Don't run migrations on dev database."
    )

    options = parser.parse_args()
    main(options)
