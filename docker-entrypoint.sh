#!/bin/sh
# Applies pending database migrations, then starts the server.
# With several replicas, set RUN_MIGRATIONS=false on all but one (or run
# `alembic upgrade head` as a separate release step) so they don't race.
set -e

# Prometheus multi-worker metrics: start from an empty directory, or counters
# from a previous run of the container would be reported again.
if [ -n "$PROMETHEUS_MULTIPROC_DIR" ]; then
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi

if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

exec "$@"
