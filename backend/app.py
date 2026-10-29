import os

# Reduce allocator fragmentation before importing torch (helps peak RSS during model load).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import threading
import uuid
import torch
import logging
import sys
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.utils import secure_filename

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

import ingest

# Set up structured JSON logging for ELK stack
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, 'service', "lawracle-backend"),
        }
        
        # Merge any extra kwargs passed to the logger (like our request details)
        if hasattr(record, 'method'):
            log_record['method'] = record.method
        if hasattr(record, 'path'):
            log_record['path'] = record.path
        if hasattr(record, 'status'):
            log_record['status'] = record.status
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        if hasattr(record, 'client_ip'):
            log_record['client_ip'] = record.client_ip
        if hasattr(record, 'query_text'):
            log_record['query_text'] = record.query_text
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
handlers = [handler]

# Ship logs directly to Logstash via TCP if available (set LOGSTASH_DISABLE=1 to skip when Logstash is down)
LOGSTASH_HOST = os.getenv('LOGSTASH_HOST', '')
LOGSTASH_PORT = int(os.getenv('LOGSTASH_PORT', '5044'))
if LOGSTASH_HOST and os.getenv('LOGSTASH_DISABLE', '').lower() not in ('1', 'true', 'yes'):
    try:
        from logstash_async.handler import AsynchronousLogstashHandler
        logstash_handler = AsynchronousLogstashHandler(
            LOGSTASH_HOST, LOGSTASH_PORT, database_path=None
        )
        # Fix: Apply the custom JSON formatter to the Logstash connection
        logstash_handler.setFormatter(JsonFormatter())
        handlers.append(logstash_handler)
    except Exception as e:
        print(f"Warning: Could not connect to Logstash at {LOGSTASH_HOST}:{LOGSTASH_PORT} - {e}")

logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# Uploads larger than this are rejected by Flask with 413 before reaching our code.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

# --- MIDDLEWARE FOR REQUEST/RESPONSE LOGGING ---
@app.before_request
def start_timer():
    # Start a stopwatch the moment the request hits the server
    g.start_time = time.time()

@app.after_request
def log_request_response(response):
    # Calculate how long the request took
    duration = time.time() - g.start_time if hasattr(g, 'start_time') else 0

    # Skip logging for the /health endpoint to avoid spamming Kibana
    if request.path == '/health':
        return response

    # Gather all the request and response details
    log_data = {
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "duration_ms": round(duration * 1000, 2),
        "client_ip": request.remote_addr,
    }

    # If you want to log the specific question the user asked
    if request.path == '/query' and request.is_json:
        try:
            # Safely try to get the query without breaking the request
            log_data["query_text"] = request.get_json(silent=True).get('query', '')
        except Exception:
            pass

    # Log as an error if it's a 4xx or 5xx status code
    if response.status_code >= 400:
        logger.error(f"{request.method} {request.path} failed", extra=log_data)
    else:
        logger.info(f"{request.method} {request.path} completed", extra=log_data)

    return response

# Environment paths for containerized deployment
DATA_DIR = os.getenv('DATA_DIR', './data')
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
# If downloading from HuggingFace, you can use "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for lightweight CPU runs.
# For cached volumes, point to the mount path.
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
# Chunks retrieved per query. 4 x 256-token chunks (see ingest.CHUNK_SIZE) plus the
# prompt and 100 new tokens fit TinyLlama's 2048-token window in a single call.
SIMILARITY_TOP_K = int(os.getenv('SIMILARITY_TOP_K', '4'))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

query_engine = None
rag_index = None
chroma_collection = None
text_splitter = None
init_error = None

# Serialises index writes (startup sync, uploads, deletes).
ingest_lock = threading.Lock()
# One model instance is shared by all request threads; run one generation at a time.
inference_lock = threading.Lock()

def truncate_text(text, max_words=200):
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + "..."

def format_leap_response(query, raw_response):
    if "Legal Issue" in raw_response and "Action Steps" in raw_response:
        if len(raw_response.split()) <= 400:
            return raw_response

    query_lower = query.lower()
    if any(word in query_lower for word in ["accident", "crash", "collision"]):
        issue = "a motor vehicle accident and potential liability under road safety laws"
    elif any(word in query_lower for word in ["property", "land", "tenant", "owner"]):
        issue = "a property dispute or real estate matter"
    elif any(word in query_lower for word in ["marriage", "divorce", "custody"]):
        issue = "a family law matter"
    else:
        issue = "a potential legal concern requiring professional advice"

    truncated_response = truncate_text(str(raw_response), 150)

    formatted_response = f"""
**Legal Issue**:
This query concerns {issue}.

**Explanation of Law**:
{truncated_response}

**Action Steps**:
1. Consult with a qualified legal professional specializing in this area of law
2. Gather all relevant documentation and evidence related to your case
3. Consider filing appropriate paperwork with relevant authorities

**Practical Guidance**:
Maintain thorough documentation of all events, communications, and expenses related to this matter.
"""
    return formatted_response

def setup_rag_system():
    """Background initialisation. Records failure so /livez can report it."""
    global init_error
    try:
        _setup_rag_system()
    except Exception as e:
        init_error = f"{type(e).__name__}: {e}"
        logger.exception("RAG pipeline initialization failed")

def _setup_rag_system():
    global query_engine, rag_index, chroma_collection, text_splitter
    logger.info("Initializing RAG pipeline...")
    # Fewer CPU threads can lower peak memory from OpenMP/BLAS during load on small nodes.
    torch.set_num_threads(int(os.getenv("TORCH_CPU_NUM_THREADS", "1")))

    system_prompt = """
You are Lawracle, an AI legal assistant specializing in Indian law.

STRICT RULES:
- Answer ONLY using information explicitly stated in the provided context documents.
- Do NOT invent, assume, or add legal provisions, punishments, or section numbers not present in the context.
- If the context lacks information, state: 'The available documents do not contain specific information on this point.'
- Always cite exact section numbers and act names as found in the context.

Use the L.E.A.P. structure:
1. Legal Issue: 1-2 sentences identifying the core legal problem.
2. Explanation of Law: 2-3 sentences citing only what the context states.
3. Action Steps: 2-3 practical steps.
4. Practical Guidance: 1-2 tips.

Keep your TOTAL response strictly under 150 words.
"""
    query_wrapper_prompt = SimpleInputPrompt(
        "<|system|>\n" + system_prompt.strip() + "</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
    )

    logger.info(f"Loading Model: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) detected! Using Metal for accelerated inference.")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="mps",
            torch_dtype=torch.float16,
        )
    else:
        logger.warning("No GPU found! Loading model on CPU with float16 to reduce memory usage.")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    gc.collect()

    embed_model = HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )

    llm = HuggingFaceLLM(
        model_name=LLM_MODEL_NAME,
        model=model,
        tokenizer=tokenizer,
        context_window=2048,
        max_new_tokens=100,
        query_wrapper_prompt=query_wrapper_prompt,
        generate_kwargs={"temperature": 0.2, "do_sample": True},
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = ingest.CHUNK_SIZE
    Settings.chunk_overlap = ingest.CHUNK_OVERLAP

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("lawracle_docs")
    logger.info(f"Opened ChromaDB collection ({collection.count()} chunks)")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    splitter = ingest.make_splitter()

    # Embed only files that are new or changed since they were last indexed.
    with ingest_lock:
        ingest.sync_index(index, collection, DATA_DIR, splitter)

    chroma_collection = collection
    rag_index = index
    text_splitter = splitter
    query_engine = index.as_query_engine(
        response_mode="compact",
        similarity_top_k=SIMILARITY_TOP_K,
    )
    logger.info("RAG pipeline ready.")

@app.route('/health', methods=['GET'])
def health():
    """Readiness: 200 only once the pipeline can answer queries."""
    if init_error:
        return jsonify({"status": "error", "model": "lawracle", "ready": False}), 500
    if query_engine is None:
        return jsonify({"status": "initializing", "model": "lawracle", "ready": False}), 503
    return jsonify({"status": "ok", "model": "lawracle", "ready": True})

@app.route('/livez', methods=['GET'])
def livez():
    """Liveness: the process is serving and initialisation has not failed.

    Stays 200 while models load, so a slow start is not mistaken for a crash,
    but turns 500 if initialisation raised, so Kubernetes restarts the pod.
    """
    if init_error:
        return jsonify({"status": "error"}), 500
    return jsonify({"status": "alive"})

@app.route('/documents', methods=['GET'])
def list_documents():
    if chroma_collection is None:
        return jsonify({"error": "Pipeline initializing"}), 503
    counts = ingest.chunk_counts(chroma_collection)
    documents = [{"source": name, "chunks": n} for name, n in sorted(counts.items())]
    return jsonify({"documents": documents, "total_chunks": sum(counts.values())})

@app.route('/documents', methods=['POST'])
def upload_document():
    """Add or replace one document. Re-uploading a file name replaces that document."""
    if query_engine is None:
        return jsonify({"error": "Pipeline initializing"}), 503

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({"error": "Send the document as multipart form field 'file'"}), 400

    filename = secure_filename(upload.filename)
    if not filename or not ingest.is_supported(filename):
        allowed = ", ".join(ingest.SUPPORTED_EXTENSIONS)
        return jsonify({"error": f"Unsupported file type; allowed: {allowed}"}), 400

    final_path = os.path.join(DATA_DIR, filename)
    # Hidden temp name: ignored by the startup sync if we crash mid-upload.
    tmp_path = os.path.join(DATA_DIR, f".upload-{uuid.uuid4().hex}.tmp")

    with ingest_lock:
        try:
            upload.save(tmp_path)
            file_hash = ingest.file_sha256(tmp_path)
            if ingest.indexed_sources(chroma_collection).get(filename) == ingest.fingerprint(file_hash):
                # Already indexed; just make sure the file itself is on disk.
                if os.path.exists(final_path):
                    os.remove(tmp_path)
                else:
                    os.replace(tmp_path, final_path)
                return jsonify({"source": filename, "status": "unchanged", "chunks": 0})

            if not ingest.read_document_text(tmp_path).strip():
                os.remove(tmp_path)
                return jsonify({"error": "No extractable text (scanned PDFs need OCR first)"}), 422

            existed = os.path.exists(final_path)
            os.replace(tmp_path, final_path)
            chunks = ingest.index_file(
                rag_index, chroma_collection, filename, final_path, text_splitter, file_hash
            )
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.exception(f"Failed to ingest uploaded file {filename}")
            return jsonify({"error": "Failed to ingest document"}), 500

    status = "updated" if existed else "added"
    logger.info(f"Document {filename} {status} ({chunks} chunks)")
    return jsonify({"source": filename, "status": status, "chunks": chunks}), 201

@app.route('/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    if chroma_collection is None:
        return jsonify({"error": "Pipeline initializing"}), 503

    # Match names exactly (existing files contain spaces), but never leave DATA_DIR.
    if os.path.basename(filename) != filename or filename.startswith(".") or "\\" in filename:
        return jsonify({"error": "Invalid document name"}), 400

    with ingest_lock:
        removed_chunks = ingest.delete_source(chroma_collection, filename)
        path = os.path.join(DATA_DIR, filename)
        file_removed = os.path.isfile(path)
        if file_removed:
            os.remove(path)

    if removed_chunks == 0 and not file_removed:
        return jsonify({"error": f"No document named {filename}"}), 404
    return jsonify({"source": filename, "removed_chunks": removed_chunks})

@app.route('/query', methods=['POST'])
def process_query():
    if not query_engine:
        return jsonify({"error": "Pipeline initializing"}), 503

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data['query']

    try:
        with inference_lock:
            raw_response = query_engine.query(query)
        formatted_response = format_leap_response(query, str(raw_response))
        return jsonify({"response": formatted_response})
    except Exception as e:
        # We don't need logger.error here since the @app.after_request middleware 
        # will catch the 500 error and log it automatically!
        return jsonify({"response": format_leap_response(query, "Unable to retrieve specific information.")}), 500

# Endpoint to capture logs sent from the frontend UI
@app.route('/log', methods=['POST'])
def receive_frontend_log():
    data = request.json
    # Log it as an error, but let Kibana know it came from the frontend UI
    logger.error(f"Frontend UI Error: {data.get('message')}", extra={"service": "lawracle-frontend-ui"})
    return jsonify({"status": "logged"})

# Start initialization in background so Gunicorn binds instantly and doesn't timeout
threading.Thread(target=setup_rag_system, daemon=True).start()

if __name__ == '__main__':
    # In production, use Gunicorn instead of app.run
    app.run(host='0.0.0.0', port=5000, use_reloader=False)