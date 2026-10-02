# Assistant Agent

An AI customer assistant for telecom billing and collections. It chats with
clients who have overdue invoices, explains their situation, and negotiates a
payment plan they can afford — served as a FastAPI backend.

What it does in a conversation:

- **Knows the client.** Pulls their invoices, payment history and line status,
  and computes a 0-100 risk score from business rules (70%) blended with a
  LightGBM model (30%).
- **Negotiates payment plans.** Offers installment options sized to that
  score. Amounts and due dates are calculated in plain Python — the language
  model never touches the money math.
- **Answers questions.** Retrieval-augmented responses from a knowledge base
  (FAISS + multilingual embeddings), with an embedded fallback.
- **Escalates problems.** Detects complaints (failed payment, identity
  verification, technical issues) and opens a support ticket with the
  client's consent.
- **Stays safe.** Guardrails on input and output block prompt injection, data
  leaks and hallucinated amounts; every endpoint is authenticated and
  rate-limited.

The demo company is the fictional "NovaTel" (set `COMPANY_NAME` to rebrand).

This is the public, portfolio version of a system originally built for a
real telecom operator. The company identity, contact details, database
schema, and client data have all been replaced with fictional/synthetic
equivalents — the scoring formula, guardrails, negotiation logic, and RAG
pipeline are faithful ports of the original.

> **Scope note:** the original system also includes an admin back-office
> (dashboards, appointment scheduling, boutique/RDV routing). That's
> standard CRUD and intentionally left out of this repo to keep the focus
> on the AI engineering: the agent loop, guardrails, hybrid scoring, and RAG.
> Support escalation is kept, simplified to a ticket log without the
> boutique-assignment subsystem.

## The problem

Collections is an inherently adversarial conversation: the company wants to
recover money, the client is often stressed or defensive, and a bad
automated response (wrong amount, tone-deaf message, or a data leak) is
worse than no automation at all. The goal here was an agent that can:

- negotiate a payment plan a client can actually afford, based on their risk profile
- never hallucinate a number — all money math is deterministic, not LLM-generated
- degrade gracefully when a dependency (retriever, LLM) isn't available, instead of failing outright
- resist prompt injection without over-blocking legitimate requests

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   FastAPI    │────▶│    Orchestrator   │────▶│   Negotiator          │
│   (main.py)  │     │  (agent loop)     │     │ (intent + negotiation)│
└─────────────┘     └─────────┬─────────┘     └───────────┬──────────┘
                               │                            │
              ┌────────────────┼────────────────┐          │
              ▼                ▼                 ▼          ▼
      ┌───────────────┐ ┌─────────────┐  ┌──────────────┐ ┌────────────────┐
      │ Hybrid Scoring │ │ RAG Retriever│  │ Output Guard │ │ Payment Planner │
      │ rules (70%) +  │ │ FAISS +      │  │ + Input Guard│ │ deterministic   │
      │ LightGBM (30%) │ │ multilingual │  │              │ │ installment math│
      │                │ │ embeddings,  │  │              │ │                 │
      │                │ │ KB fallback  │  │              │ │                 │
      └───────────────┘ └─────────────┘  └──────────────┘ └────────────────┘
              │
              ▼
      ┌───────────────┐
      │ Short/long-term│
      │ memory (session│
      │ + history)     │
      └───────────────┘
```

## Engineering decisions worth highlighting

**The LLM never computes money.** `assistant/tools/payment_planner.py` is
pure, deterministic Python — installment amounts, due dates, and rounding
are all calculated in code and unit-tested directly. The LLM only explains
a plan this module already built. This is the single most important
guardrail in the system: an LLM inventing a payment figure is a much worse
failure mode than a slightly-off chatbot reply.

**Hybrid scoring, not pure ML.** `recouvrement/scoring.py` computes a
rules-based score (new clients get a neutral 70; existing clients start at
50 and earn up to 50 points for on-time payment history, then lose points
for late payments, unpaid balance, past suspensions, and past
deactivations). `recouvrement/predict.py` blends that with a LightGBM
model, 70/30. A pure black-box score is hard to justify to a client or an
auditor in a regulated market like consumer credit; blending in
explainable rules is standard practice in real collections/risk systems.

**RAG degrades gracefully.** `assistant/rag/retriever.py` exposes an
`is_ready` check the orchestrator consults before every query. If the
FAISS index hasn't been built yet (or the embedding model couldn't be
downloaded — see Setup below), the orchestrator falls back to a small
embedded knowledge base instead of crashing.

**Multilingual, multi-format RAG.** The knowledge base can ingest PDF,
DOCX, Excel, CSV, JSON, Markdown, and plain text, and embeds everything
with a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`, FR/AR/EN)
into a FAISS index — `IndexFlatIP` under 10k chunks, `IndexIVFFlat` above
that for scale.

**Guardrails check danger before topic.** `output_guard.py` validates in
this order: data leak -> prompt injection -> unrealistic amount ->
off-topic. An earlier version checked "is this on-topic" first, which
caused legitimate but unusually-phrased responses to get false-positive
blocked. Reordering the checks fixed the false-positive rate without
weakening any of the actual security checks.

**Two-layer intent classification.** A fast local keyword layer resolves
most messages with zero LLM calls; only genuinely ambiguous messages fall
through to an LLM classification call. This keeps both latency and
per-conversation cost down.

**A real bug, found and fixed during this rebuild.** The chunking
function shared by every document loader had a termination bug: once a
section's remaining text was shorter than the overlap window, `end -
overlap` fell behind `start`, so the loop kept re-chunking the same short
tail with a 1-character-shrinking window instead of stopping. A 2KB FAQ
file was exploding into 600+ near-duplicate chunks. Fixed by breaking the
loop once a chunk reaches the end of the text (`assistant/rag/document_loader.py`).
Left in as a reminder that "it ran without crashing" isn't the same as "it's
correct" — this bug never raised an exception, it just silently bloated the
index by two orders of magnitude.

## Results

**ML risk model** (LightGBM, trained on the public credit-score dataset,
13 features — 8 raw + 5 engineered: `score_retard`, `ratio_dette_anciennete`,
`risque_global`, `charge_credit`, `complexite_financiere` — see `recouvrement/train.py`):

| Metric | Value |
|---|---|
| Accuracy | 67.1% |
| Precision | 57.1% |
| Recall | 66.0% |
| F1 | 61.3% |
| AUC-ROC | 73.8% |
| 5-fold CV F1 | 58.6% +/- 1.3% |

Full confusion matrix and feature importance charts: `reports/confusion_matrix.png`, `reports/feature_importance.png`.

> **Reproducing these numbers:** the metrics above come from training on a
> cleaned copy of the public credit-score dataset (`data/credit_score_cleaned.csv`),
> which is not committed to this repo. On a fresh clone, `recouvrement/train.py`
> finds no dataset and generates a synthetic one instead, so the metrics you
> (and CI) get will differ. To reproduce them, place the cleaned CSV at
> `data/credit_score_cleaned.csv` before training.

**Guardrail adversarial eval** (`scripts/eval_guardrails.py`, 32 adversarial
cases — prompt injection, role override, jailbreak attempts, data-leak
probing, unrealistic amounts, genuine off-topic messages):

| Guard | Pass rate |
|---|---|
| Input sanitization | 18/18 (100%) |
| Output validation | 14/14 (100%) |

**Test suite:** 152 tests across guardrails, intent classification,
payment-plan math, authentication and authorization, session ownership,
rate limiting, both session backends, the data-access queries, plan
confirmation idempotency, log privacy, LLM time budgets and monitoring
(`pytest tests/ -v`). CI runs the suite twice: once on SQLite with in-memory
sessions, once on Postgres + Redis.

## Running it

```bash
pip install -r requirements-dev.txt   # runtime + test/training tools
cp .env.example .env                  # then set JWT_SECRET (see below)
python seed.py                        # applies migrations + 25 synthetic clients
python recouvrement/train.py          # trains the LightGBM model
uvicorn main:app --reload
```

Every endpoint except `/health` and `/ready` needs a bearer token. Locally,
mint one with the dev helper (it signs with your `JWT_SECRET`):

```bash
TOKEN=$(python scripts/issue_token.py --client-id 3)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/client/assistant/auto-session/3
```

That's a fully working agent. Semantic RAG search is optional — it needs
FAISS + a multilingual embedding model, which pulls in torch (~2GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, much smaller
pip install -r requirements-rag.txt
python scripts/build_index.py     # downloads the embedding model + builds the FAISS index
```

Or the production-like stack with Docker (API + Postgres + Redis):

```bash
cp .env.example .env                              # set JWT_SECRET and POSTGRES_PASSWORD
docker compose up --build -d
docker compose run --rm agent python seed.py      # optional demo data
```

Two things that fail *gracefully* rather than breaking the app:

- **No `GROQ_API_KEY`:** every endpoint still runs end-to-end - scoring,
  RAG retrieval, and payment-plan generation all work fully offline; only
  the free-text LLM reply falls back to a deterministic message. Responses
  served this way carry `"degraded": true` so a caller can tell, and the
  amounts in them still come from the payment planner, never from a model.
  No network call is attempted when the key is missing. Copy `.env.example`
  to `.env` and add a key to enable live LLM responses.
- **RAG stack not installed at all:** faiss, sentence-transformers, torch,
  pypdf and python-docx are imported lazily, so the API server boots and
  every endpoint works with only `requirements.txt` installed. The retriever
  simply reports `is_ready == False` and the orchestrator uses its embedded
  knowledge base instead.
- **No internet access to Hugging Face:** `scripts/build_index.py` downloads
  a ~120MB embedding model on first run. If that fails (e.g. a
  network-restricted CI environment), the script prints a clear message
  and exits non-zero - same fallback applies.

## Security model

- **Authentication.** The API doesn't own user accounts. It verifies JWTs
  issued by the operator's identity provider: HS256 with `JWT_SECRET`, or
  RS256/ES256 with `JWT_PUBLIC_KEY`, plus optional `JWT_ISSUER` /
  `JWT_AUDIENCE` checks. Tokens must carry `exp`; `alg: none` and
  unlisted algorithms are rejected.
- **Authorization.** A client token (`sub` = client id, `role` = `client`)
  can only act on that client. `client_id` in request bodies is optional and
  must match the token if sent. Staff tokens (`role` = `staff`) can read any
  client's risk score (`/scoring`, `/ml/score-combine`) but can't chat or
  confirm plans for a client.
- **Session ownership.** Each session records its client. Using another
  client's `session_id` returns 403. Session ids are full random UUIDs.
- **Rate limiting.** `/chat` and `/option-click` (the endpoints that can call
  the paid LLM) are limited per client (`LLM_RATE_LIMITS`, default
  `20/minute;300/day`) and return 429 with `Retry-After`. Messages are capped
  at `MAX_MESSAGE_LENGTH` characters.
- **CORS.** Only origins in `CORS_ALLOWED_ORIGINS` are allowed; credentials
  mode is off, since tokens go in the `Authorization` header.
- **One plan per client.** Confirming is idempotent: a double click, a
  retried request or a second session can't create a second plan. The
  session claim, a database check and a unique index on confirmed plans
  each enforce it.
- **No personal data in logs.** Logs record ids, never what clients typed;
  email addresses and phone numbers are masked (`a***@example.com`), and SQL
  errors hide their parameters in production.

## Deploying

Set `APP_ENV=production` (the Docker image does). The app then refuses to
start unless it has Postgres, `REDIS_URL`, a JWT key (a secret of 32+
characters for HS256), and CORS origins without `*`.

- **Database migrations** use Alembic (`migrations/`). The container runs
  `alembic upgrade head` on start. With several replicas, set
  `RUN_MIGRATIONS=false` on all but one, or run it as a separate release
  step. To change the schema, edit `models.py`, then
  `alembic revision --autogenerate -m "..."`, review the file, and commit it.
  CI fails if the models and migrations drift apart (`alembic check`).
- **Sessions and rate limits** live in Redis, so they're shared across
  workers and replicas and survive restarts. Without `REDIS_URL` (dev only)
  they're kept in process memory, and expired sessions are purged.
- **Image.** A multi-stage build: compilers stay in the build stage, the
  runtime image has no test tools or demo data, runs as a non-root user,
  starts `WEB_CONCURRENCY` uvicorn workers (default 2), and has a
  `HEALTHCHECK` on `/health`. Use `/ready` (database + Redis) for your
  orchestrator's readiness probe. Behind a load balancer, set
  `FORWARDED_ALLOW_IPS` to its address.
- **Model.** Put the `model_*.pkl` files trained on real data in
  `recouvrement/` before `docker build`. Without them, the build trains a
  model on synthetic data, which is only good enough for a demo.
- **LLM latency** is capped: each LLM call gets `LLM_TOTAL_TIMEOUT_SECONDS`
  (default 15) across all retries, intent classification 4. When the budget
  runs out the client gets the deterministic reply instead of waiting.
  Confirmation emails are sent in the background, so a slow mail server
  never delays a response.

## Monitoring

- **Logs** are JSON lines in production (`LOG_FORMAT=json`), one access line
  per request with method, route, status and duration. Every line carries a
  request id, also returned as the `X-Request-ID` response header. A valid
  incoming `X-Request-ID` (from your load balancer) is reused.
- **Metrics** at `GET /metrics` for Prometheus, enabled by setting
  `METRICS_TOKEN` and scraped with `Authorization: Bearer <METRICS_TOKEN>`.
  Aggregated across all workers (`PROMETHEUS_MULTIPROC_DIR`, set in the
  image). Includes `http_requests_total`, `http_request_duration_seconds`,
  `llm_requests_total{outcome}`, `llm_request_duration_seconds`,
  `degraded_replies_total`, `guardrail_blocks_total`,
  `rate_limited_requests_total` and `payment_plans_confirmed_total`.
- **Errors** go to Sentry when `SENTRY_DSN` is set, without request bodies,
  local variables or user details.
- **Alerts** worth setting up in Prometheus/Alertmanager or your provider:
  5xx rate above 1% of `http_requests_total`; p95 of
  `http_request_duration_seconds` above 10s on `/client/assistant/chat`;
  `llm_requests_total{outcome="error"}` above 20% of LLM calls (users are
  getting fallback replies); `/ready` failing; a jump in
  `guardrail_blocks_total` (someone probing the agent).

## Stack

Python, FastAPI, SQLAlchemy, Alembic, Postgres/SQLite, Redis, PyJWT, Prometheus, Sentry, LightGBM,
scikit-learn, sentence-transformers, FAISS, Groq (Llama 3.3), pytest, Docker,
GitHub Actions.
