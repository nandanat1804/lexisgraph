# CPU-only image, no GPU drivers needed
FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf/torch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheel (much smaller than default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the small embedding/reranker models at build time so the
# container doesn't need HuggingFace access at runtime (optional but
# recommended for production - comment out if you'd rather download lazily)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

# Bake the sample docs into the index at build time so the API is
# queryable immediately; re-ingest via POST /ingest for real documents
RUN python ingest.py --docs sample_docs

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
