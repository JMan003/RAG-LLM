# Lawracle – AI Legal Assistant (RAG-LLM)

Lawracle is an AI-powered legal assistant specializing in Indian law. It utilizes **Mistral-7B / TinyLlama** combined with **LlamaIndex** and **ChromaDB** for Retrieval-Augmented Generation (RAG).

---

## How to Start Locally

You can run this project locally using either **Docker Compose** (recommended) or **Natively**.

### Option 1: Using Docker Compose (Recommended)

This method spins up both the Python Flask backend and the Nginx frontend in containerized environments. It automatically handles URL proxying exactly as configured for production.

1. **Start the containers**:
   ```bash
   docker compose up --build
   ```
2. **Access the application**:
   - Open your browser and navigate to: **[http://localhost:8081](http://localhost:8081)**
3. **Adding documents**:
   - Place your PDF, TXT, or MD documents into the `./data` directory. On startup the backend indexes only files that are new or changed since the last run (tracked by SHA-256 hash), or upload them while it runs (see [Managing documents](#managing-documents)).

*Note: The first startup will download the `TinyLlama` model (~2.2GB), which will be cached in a Docker volume for subsequent starts.*

---

### Option 2: Running Natively

If you prefer not to use Docker, you can run the Flask backend and serve the frontend directly.

#### 1. Start the Flask Backend
1. Make sure you have **Python 3.10+** installed.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the backend:
   ```bash
   python app.py
   ```
   *The backend will start running on [http://127.0.0.1:5000](http://127.0.0.1:5000).*

#### 2. Open the Frontend
1. Double-click the `frontend/index.html` file to open it directly in your browser (`file:///...`), or serve it using a lightweight server (e.g. `npx serve` or `python -m http.server`).
2. The frontend is configured to automatically route API requests to `http://127.0.0.1:5000` when running locally.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Ask a question: `{"query": "..."}` |
| `GET` | `/health` | Readiness: `200` once models are loaded and the index is synced, `503` while starting, `500` if startup failed |
| `GET` | `/livez` | Liveness: `200` while the process is healthy (including while loading), `500` if startup failed |
| `GET` | `/documents` | List indexed documents and their chunk counts |
| `POST` | `/documents` | Upload a PDF/TXT/MD file (multipart field `file`, max 25 MB); re-uploading a name replaces it |
| `DELETE` | `/documents/<name>` | Remove a document's chunks and file |

Through the frontend or ingress, prefix these paths with `/api`.

## Managing documents

Every chunk stores its source file name and the file's SHA-256 hash. At startup the backend compares the
data directory with the index and embeds only new or changed files, so restarts don't re-embed everything
and new files are never silently ignored. Indexed documents whose file is missing are kept (and logged),
never deleted automatically; remove them explicitly with `DELETE /documents/<name>`.

In Kubernetes, documents live on the `lawracle-data-pvc` volume. Seed it by uploading the bundled acts:

```bash
for f in data/*.pdf; do curl -sS -F "file=@$f" http://lawracle.test/api/documents; echo; done
```

Uploaded file names are sanitised (spaces become underscores).

## Evaluating retrieval

`backend/evaluation/questions.jsonl` holds 34 questions covering all 22 acts. Each names the act that
answers it and a phrase from the relevant section. The script reports, for each k:

- **document hit@k**: a chunk from the right act is in the top k
- **chunk hit@k**: that chunk also contains the expected phrase

It also reports MRR, retrieval latency (p50/p95), and the questions whose act is missing from the app's
top `SIMILARITY_TOP_K` results.

```bash
# Docker Compose
docker compose exec backend python evaluation/eval_retrieval.py --top-k 8

# Kubernetes
kubectl exec deploy/lawracle-backend -- python evaluation/eval_retrieval.py --json-out /tmp/eval.json
```

Use it to compare settings before changing them.

## Retrieval settings

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `256` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Tokens shared between neighbouring chunks |
| `SIMILARITY_TOP_K` | `4` | Chunks retrieved per query |

Each chunk records the chunk settings it was built with, so changing `CHUNK_SIZE` or `CHUNK_OVERLAP`
re-embeds every document once on the next startup. Keep `CHUNK_SIZE x SIMILARITY_TOP_K` around 1024 so
the retrieved text, prompt and answer fit TinyLlama's 2048-token window in one generation call.

Baseline on the 34-question set with the previous settings (512 tokens, top 2): the right act was in the
top 2 for 97.1% of questions, but the exact section only 55.9% of the time (73.5% at top 4).

## Tests

```bash
cd backend && pip install -r requirements-test.txt && PYTHONPATH=. pytest tests/
```

Jenkins runs the same suite inside the built image before pushing it.

