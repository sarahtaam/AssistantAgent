# syntax=docker/dockerfile:1

# ── 1. Build dependencies (compilers stay in this stage only) ─────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── 2. Model artifacts ─────────────────────────────────────────────────────
# Production should ship a model trained on real data: put model_*.pkl in
# recouvrement/ before building and they are used as-is. Otherwise a model
# is trained here on the synthetic fallback dataset (demo quality only).
FROM builder AS model

RUN pip install matplotlib==3.9.2
WORKDIR /build
COPY recouvrement/ recouvrement/
RUN mkdir -p data reports \
    && if [ ! -f recouvrement/model_lgbm.pkl ]; then \
         echo "WARNING: no model artifacts provided - training on synthetic data." \
         && python recouvrement/train.py; \
       fi

# ── 3. Runtime image ───────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp: OpenMP runtime LightGBM links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    WEB_CONCURRENCY=2 \
    RUN_MIGRATIONS=true \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . .
COPY --from=model --chown=app:app /build/recouvrement/*.pkl recouvrement/
RUN mkdir -p data && chown app:app data

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
# --proxy-headers trusts X-Forwarded-* only from FORWARDED_ALLOW_IPS (your load balancer).
# Access logging is done by the app (JSON, with request ids), not uvicorn.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1} --no-server-header --no-access-log"]
