# CPU-only image, no GPU drivers needed. No torch here at all - embeddings
# and reranking run on ONNX runtime (via fastembed), which keeps the whole
# image AND runtime memory usage small enough for free-tier hosts.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the small embedding/reranker ONNX models at build time so
# the container doesn't need HuggingFace access at runtime, and so the
# models aren't loaded into memory twice (once at build, once at first
# request) - this download only happens once, here.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

# Bake the sample docs into the index at build time so the API is
# queryable immediately; re-ingest via POST /ingest for real documents
RUN python ingest.py --docs sample_docs

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
