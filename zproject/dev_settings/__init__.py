# Development environment overrides
# This file contains settings overrides for the development environment

DEVELOPMENT = True
DEBUG = True
SERVE_STATIC_FILES = True

# Required settings that Zulip expects
EXTERNAL_HOST = 'localhost:8000'
ZULIP_ADMINISTRATOR = 'admin@example.com'

# Ensure static files are properly served in development
STATICFILES_DIRS = [
    "/home/ubuntu/gpChat/static/",
]

# Make sure templates are cached properly
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.jinja2.Jinja2',
        'APP_DIRS': True,
        'OPTIONS': {
            'debug': DEBUG,
        },
    },
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'APP_DIRS': True,
        'OPTIONS': {
            'debug': DEBUG,
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
            ],
        },
    },
]
