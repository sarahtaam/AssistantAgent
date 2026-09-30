#!/bin/sh
# Applies pending database migrations, then starts the server.
# With several replicas, set RUN_MIGRATIONS=false on all but one (or run
# `alembic upgrade head` as a separate release step) so they don't race.
set -e

if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

exec "$@"
