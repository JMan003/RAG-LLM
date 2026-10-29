"""Incremental document ingestion for the Lawracle RAG backend.

Every chunk stored in ChromaDB carries three metadata fields:
  - ``source``:       the file name the chunk came from
  - ``file_hash``:    SHA-256 of that file's bytes when it was indexed
  - ``chunk_config``: the chunking settings used, e.g. "256/50"

On startup (and on upload) we compare the files in the data directory with
what is already indexed and only embed what is new or changed, instead of
indexing everything once and then ignoring the data directory forever.
Changing CHUNK_SIZE or CHUNK_OVERLAP re-embeds every document once, because
their stored chunk_config no longer matches.

Indexed sources whose file is missing from the data directory are *kept* and
logged, never deleted automatically: removing documents is an explicit action
(``DELETE /documents/<name>``), so an empty or unmounted data volume can never
wipe the index.

Heavy libraries (llama_index, PyMuPDF) are imported lazily so the planning
logic can be unit-tested without them.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
SOURCE_KEY = "source"
HASH_KEY = "file_hash"
CONFIG_KEY = "chunk_config"
# Tuned with evaluation/eval_retrieval.py: 256-token chunks with the top 4
# retrieved fill the same context as 512 x 2 but find the right section more often.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_CONFIG = f"{CHUNK_SIZE}/{CHUNK_OVERLAP}"
# Bookkeeping fields that must not be embedded or shown to the LLM.
INTERNAL_KEYS = [HASH_KEY, CONFIG_KEY]


@dataclass
class SyncPlan:
    """What a sync needs to do, computed purely from file and index hashes."""

    add: list = field(default_factory=list)        # file present, never indexed
    update: list = field(default_factory=list)     # file present, hash differs or unknown
    unchanged: list = field(default_factory=list)  # file present, hash matches
    orphaned: list = field(default_factory=list)   # indexed, but no file in data dir


def is_supported(filename):
    return (
        not filename.startswith(".")
        and filename.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def file_sha256(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def list_data_files(data_dir):
    """Return {filename: path} for supported, non-hidden files in data_dir."""
    if not os.path.isdir(data_dir):
        return {}
    files = {}
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and is_supported(name):
            files[name] = path
    return files


def fingerprint(file_hash, chunk_config=None):
    """What must match for a file's chunks to be reusable: content and chunking."""
    return f"{file_hash}|{chunk_config or CHUNK_CONFIG}"


def plan_sync(current, indexed):
    """Compare {filename: fingerprint} on disk with the index.

    ``indexed`` maps filename to a fingerprint, or ``None`` for chunks written
    before hashes and chunk settings were recorded; those files are re-indexed
    once so they gain both.
    """
    plan = SyncPlan()
    for name in sorted(current):
        if name not in indexed:
            plan.add.append(name)
        elif indexed[name] != current[name]:
            plan.update.append(name)
        else:
            plan.unchanged.append(name)
    plan.orphaned = sorted(name for name in indexed if name not in current)
    return plan


def indexed_sources(collection):
    """Return {source: fingerprint or None} for everything in the collection.

    The fingerprint is ``None`` when chunks lack a hash or chunk settings
    (written by an older version), which forces a one-off re-index.
    """
    result = collection.get(include=["metadatas"])
    sources = {}
    for metadata in result.get("metadatas") or []:
        if not metadata or SOURCE_KEY not in metadata:
            continue
        name = metadata[SOURCE_KEY]
        file_hash = metadata.get(HASH_KEY)
        chunk_config = metadata.get(CONFIG_KEY)
        value = fingerprint(file_hash, chunk_config) if file_hash and chunk_config else None
        # Keep a known fingerprint if any chunk of this source has one.
        if sources.get(name) is None:
            sources[name] = value
    return sources



def chunk_counts(collection):
    """Return {source: number_of_chunks} for everything in the collection."""
    result = collection.get(include=["metadatas"])
    counts = {}
    for metadata in result.get("metadatas") or []:
        if metadata and SOURCE_KEY in metadata:
            name = metadata[SOURCE_KEY]
            counts[name] = counts.get(name, 0) + 1
    return counts


def read_document_text(path):
    """Extract text from a PDF, TXT or MD file. Returns '' if nothing usable."""
    if path.lower().endswith(".pdf"):
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return "".join(page.get_text() for page in doc)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def make_splitter():
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.utils import globals_helper

    # Measure chunks with LlamaIndex's default tiktoken tokenizer explicitly, so
    # chunking stays stable (and matches the existing index) even when
    # Settings.tokenizer is set to the LLM's tokenizer for prompt packing.
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=globals_helper.tokenizer,
    )


def build_nodes(source, text, file_hash, splitter):
    from llama_index.core import Document

    document = Document(
        text=text,
        metadata={SOURCE_KEY: source, HASH_KEY: file_hash, CONFIG_KEY: CHUNK_CONFIG},
        # Bookkeeping only: keep it out of embeddings and LLM prompts.
        excluded_embed_metadata_keys=INTERNAL_KEYS,
        excluded_llm_metadata_keys=INTERNAL_KEYS,
    )
    return splitter.get_nodes_from_documents([document])


def delete_source(collection, source):
    """Delete every chunk of one source. Returns the number of chunks removed."""
    existing = collection.get(where={SOURCE_KEY: source})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def index_file(index, collection, source, path, splitter, file_hash=None):
    """(Re)index one file. Returns the number of chunks inserted (0 if no text)."""
    file_hash = file_hash or file_sha256(path)
    text = read_document_text(path)
    if not text.strip():
        logger.warning(f"No extractable text in {source}; skipping")
        return 0
    nodes = build_nodes(source, text, file_hash, splitter)
    # Remove the old version only once the new one is ready to insert.
    delete_source(collection, source)
    index.insert_nodes(nodes)
    return len(nodes)


def sync_index(index, collection, data_dir, splitter):
    """Bring the index up to date with data_dir. Returns a summary dict."""
    files = list_data_files(data_dir)
    hashes = {name: file_sha256(path) for name, path in files.items()}
    current = {name: fingerprint(h) for name, h in hashes.items()}
    plan = plan_sync(current, indexed_sources(collection))
    logger.info(f"Index sync with chunk settings {CHUNK_CONFIG}")

    chunks_added = 0
    skipped = []
    for name in plan.add + plan.update:
        try:
            added = index_file(index, collection, name, files[name], splitter, hashes[name])
        except Exception:
            logger.exception(f"Failed to index {name}")
            skipped.append(name)
            continue
        if added == 0:
            skipped.append(name)
        chunks_added += added
        logger.info(f"Indexed {name} ({added} chunks)")

    if plan.orphaned:
        logger.warning(
            f"{len(plan.orphaned)} indexed source(s) have no file in {data_dir} "
            f"and were kept: {plan.orphaned}"
        )

    summary = {
        "added": plan.add,
        "updated": plan.update,
        "unchanged": len(plan.unchanged),
        "orphaned": plan.orphaned,
        "skipped": skipped,
        "chunks_added": chunks_added,
    }
    logger.info(f"Index sync complete: {summary}")
    return summary
