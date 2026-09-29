# NovaTel Collections Agent

A conversational AI agent that negotiates payment plans with overdue telecom
clients — combining a rules+ML hybrid risk score, retrieval-augmented
responses, and LLM guardrails that keep a language model from ever touching
the actual money math.

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

**Test suite:** 44 tests across guardrails, intent classification, and
payment-plan math (`pytest tests/ -v`).

## Running it

```bash
pip install -r requirements.txt   # core deps only (~200MB)
python seed.py                    # generates 25 synthetic clients
python recouvrement/train.py      # trains the LightGBM model
uvicorn main:app --reload
```

That's a fully working agent. Semantic RAG search is optional — it needs
FAISS + a multilingual embedding model, which pulls in torch (~2GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, much smaller
pip install -r requirements-rag.txt
python scripts/build_index.py     # downloads the embedding model + builds the FAISS index
```

Or with Docker (does all of the above at build time):

```bash
docker compose up --build
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

## Stack

Python, FastAPI, SQLAlchemy, SQLite, LightGBM, scikit-learn, sentence-transformers,
FAISS, Groq (Llama 3.3), pytest, Docker, GitHub Actions.
