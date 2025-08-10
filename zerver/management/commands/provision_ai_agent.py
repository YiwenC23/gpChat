from django.core.management.base import BaseCommand

from zerver.lib.agent_provision import provision_ai_agent_for_realm
from zerver.models import Realm
from zerver.models.realms import get_realm


class Command(BaseCommand):
    help = "Provisions AI agent bots for specified realm or all realms, and seeds AI model configuration."

    def add_arguments(self, parser):
        parser.add_argument(
            "realm",
            nargs="?",
            help="The realm to provision the bot in. Omit when using --all-realms.",
        )
        parser.add_argument(
            "--all-realms",
            action="store_true",
            help="Provision AI agent bots for all existing realms.",
        )
        parser.add_argument(
            "--name", default="AI Agent", help="The full name of the bot."
        )
        parser.add_argument(
            "--email", default="ai-agent-bot", help="The email prefix of the bot."
        )
        parser.add_argument(
            "--seed-config",
            action="store_true",
            default=True,
            help="Also seed AI model configuration (default: True).",
        )

    def handle(self, *args, **options):
        all_realms = options.get("all_realms", False)
        realm_name = options.get("realm")
        name = options["name"]
        email_prefix = options["email"]
        seed_config = options["seed_config"]

        # Validate arguments
        if all_realms and realm_name:
            self.stderr.write(
                self.style.ERROR(
                    "Cannot specify both a realm and --all-realms. Choose one."
                )
            )
            return

        if not all_realms and not realm_name:
            self.stderr.write(
                self.style.ERROR(
                    "Must specify either a realm name or --all-realms."
                )
            )
            return

        # Get realms to process
        if all_realms:
            realms = Realm.objects.filter(deactivated=False)
            if not realms.exists():
                self.stderr.write(self.style.ERROR("No active realms found."))
                return
            self.stdout.write(
                f"Provisioning AI agent bots for {realms.count()} active realm(s)..."
            )
        else:
            try:
                realms = [get_realm(realm_name)]
            except Realm.DoesNotExist:
                self.stderr.write(
                    self.style.ERROR(f"Realm '{realm_name}' does not exist.")
                )
                return

        # Process each realm
        for realm in realms:
            self.stdout.write(f"\nProcessing realm: {realm.string_id}")

            # Use the library function to provision the AI agent
            bot_profile = provision_ai_agent_for_realm(
                realm=realm,
                bot_name=name,
                email_prefix=email_prefix,
                seed_config=seed_config,
            )

            if bot_profile:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  Successfully created AI agent bot '{name}' with API key '{bot_profile.api_key}'"
                    )
                )
            else:
                self.stdout.write(f"  AI agent bot already exists or could not be created")

        self.stdout.write(self.style.SUCCESS("\nDone!"))
