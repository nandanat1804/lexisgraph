---
title: LexisGraph
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# LexisGraph
### CPU-Friendly Enterprise Legal RAG with Hybrid Retrieval

A legal document Q&A system built for a **non-gaming laptop** — no GPU,
no Docker requirement, no paid vector-DB server. It combines semantic
(vector) search with keyword (BM25) search via Reciprocal Rank Fusion,
reranks with a small cross-encoder, enriches with a lightweight
knowledge graph, and generates cited answers through an LLM API call.

```
Documents → preprocessing → chunking → embeddings → Qdrant (vector) + BM25 (keyword)
   → RRF fusion → cross-encoder reranking → knowledge graph enrichment
   → LLM (Anthropic/OpenAI/Groq) → cited answer → FastAPI
```

## Why this runs on a laptop with no GPU

| Original heavy component | This version |
|---|---|
| ColPali (vision-language retrieval) | Text-only extraction (pypdf) — no GPU model |
| Fine-tuned/self-hosted LLM | API call (Anthropic / OpenAI / Groq) — inference happens on their servers |
| Neo4j server (Docker) | `networkx` in-memory/on-disk graph — same graph concepts, zero server |
| Large embedding model | `all-MiniLM-L6-v2` (~80MB, CPU inference in milliseconds) |
| Qdrant server (Docker) | Qdrant **embedded mode** — same client API, runs as a local library |

Everything that *does* run locally (embeddings, BM25, reranking,
graph traversal) uses models small enough to run comfortably on CPU.

## 1. Setup

```bash
# from the lexisgraph/ directory
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> First run of the embedding/reranker models will download them from
> HuggingFace (~200MB total, one-time, then cached locally forever).

Copy the environment template and add **one** LLM API key:

```bash
cp .env.example .env
```

Edit `.env`:
- Set `LLM_PROVIDER` to `anthropic`, `openai`, or `groq`
- Fill in the matching `*_API_KEY`

Don't have a key yet? Leave `LLM_PROVIDER=none` — retrieval still works
and you'll get the raw retrieved passages instead of a generated answer,
so you can test the whole pipeline before wiring up an LLM.

## 2. Ingest documents

A sample legal document (`sample_docs/sample_service_agreement.txt`) is
included so you can test immediately:

```bash
python ingest.py --docs sample_docs
```

To use your own documents, drop `.pdf` or `.txt` files into a folder
and point `--docs` at it:

```bash
python ingest.py --docs /path/to/your/contracts
```

This builds three local indexes under `./data/`:
- `qdrant_db/` — vector index (embedded Qdrant)
- `bm25_index.pkl` — keyword index
- `knowledge_graph.gpickle` — entity graph (parties, sections, dates, amounts, citations)

## 3. Query

**Interactive CLI:**
```bash
python query_cli.py
```

**Single question:**
```bash
python query_cli.py --question "How much notice is required to terminate for convenience?"
```

**REST API:**
```bash
uvicorn api:app --reload --port 8000
```
Then open `http://localhost:8000/docs` for interactive Swagger UI, or:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the liability cap?", "top_k": 5}'
```

## 4. Evaluate

```bash
python run_evaluation.py
```

Reports **Hit Rate@k**, **MRR** (retrieval quality) and **ROUGE-L**
(generation quality vs. reference answers) — edit `run_evaluation.py`
to add your own eval questions.

## Project layout

```
lexisgraph/
├── src/
│   ├── config.py              # env-based configuration
│   ├── document_processor.py  # PDF/TXT extraction + chunking
│   ├── embeddings.py          # sentence-transformers wrapper (CPU)
│   ├── vector_store.py        # Qdrant embedded-mode wrapper
│   ├── bm25_search.py         # BM25 keyword search
│   ├── reranker.py            # cross-encoder reranking
│   ├── hybrid_retrieval.py    # RRF fusion + reranking orchestration
│   ├── knowledge_graph.py     # networkx KG + legal entity extraction
│   ├── llm_client.py          # pluggable Anthropic/OpenAI/Groq calls
│   ├── rag_pipeline.py        # ties every stage together
│   └── evaluation.py          # Hit Rate, MRR, ROUGE-L
├── ingest.py                  # CLI: build indexes from a doc folder
├── query_cli.py                # CLI: ask questions interactively
├── api.py                     # FastAPI server
├── run_evaluation.py           # example eval run
├── sample_docs/                # a sample service agreement to test with
├── requirements.txt
└── .env.example
```

## Securing the API before deploying publicly

`api.py` supports an optional API key. If `API_KEY` is left blank in
`.env`, auth is skipped (fine for localhost-only use). Before deploying
anywhere with a public IP, set it:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # generate one
```
Paste the result into `API_KEY=` in `.env`, then send it as a header:
```bash
curl -X POST http://your-server:8000/query \
  -H "X-API-Key: your-generated-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "..."}'
```
`/health` stays open (no key needed) for uptime checks; `/status`,
`/ingest`, and `/query` all require the key once one is set.

## Using real Neo4j instead of the built-in graph (optional)

The default knowledge graph uses `networkx` so nothing extra needs to
run. If you want to demo real Cypher queries against Neo4j instead:

1. `docker run -p 7474:7474 -p 7687:7687 neo4j:latest`
2. `pip install neo4j`
3. In `.env`: `USE_NEO4J=true`, plus `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
4. Use `knowledge_graph.neo4j_adapter()` from `src/knowledge_graph.py` to get a driver

This is entirely optional — the pipeline works identically either way.

## Extending

- **More document types:** add a loader function in `document_processor.py`
- **Different embedding model:** change `EMBEDDING_MODEL` in `.env` (any
  sentence-transformers model works, just check its size vs. your RAM)
- **Local LLM instead of API:** swap `llm_client.py`'s `generate_answer`
  to call `ollama` (e.g. `llama3.1:8b-instruct-q4`) if you'd rather run
  fully offline and have ~8GB RAM to spare — slower on CPU than the API
  route but zero cost and no internet needed at query time
