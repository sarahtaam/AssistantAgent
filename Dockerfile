FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional RAG extras are deliberately NOT installed: they add ~2GB (torch)
# and the agent runs fully without them via its knowledge-base fallback.
# To enable semantic search, also COPY requirements-rag.txt, pip install it,
# and run scripts/build_index.py below.

COPY . .

RUN mkdir -p data reports

# Seed the demo DB and train the model at image build time so `docker run`
# gives a fully working demo with zero manual steps.
RUN python seed.py \
    && python recouvrement/train.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
