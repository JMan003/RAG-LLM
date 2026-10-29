"""Measure retrieval quality and latency against a labelled question set.

Each line of questions.jsonl has:
  question         the user query
  expected_source  file name of the act that answers it
  must_contain     a short phrase from the relevant section

Two hit levels are reported:
  document hit  a retrieved chunk comes from expected_source
  chunk hit     that chunk also contains must_contain (stricter)

Run it where the index lives, e.g. inside the backend container:
    python evaluation/eval_retrieval.py
    python evaluation/eval_retrieval.py --top-k 8 --json-out results.json

The app currently answers with the top 2 chunks (similarity_top_k=2), so
hit@2 is the number that reflects what users get today.
"""

import argparse
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_TOP_K = 2


def normalize(text):
    """Lower-case, unify quotes and collapse whitespace so PDF line breaks don't matter."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def source_key(name):
    """Compare file names the way uploads store them.

    POST /documents sanitises names with werkzeug's secure_filename
    ("IT Act.pdf" is stored as "IT_Act.pdf"), while files indexed from the
    data directory keep their original names. Normalising both sides makes
    the evaluation correct either way.
    """
    from werkzeug.utils import secure_filename

    return secure_filename(name or "").lower()


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(retrieved, expected_source, must_contain=None):
    """1-based rank of the first matching chunk in [(source, text), ...], or None."""
    phrase = normalize(must_contain) if must_contain else None
    expected = source_key(expected_source)
    for rank, (source, text) in enumerate(retrieved, start=1):
        if source_key(source) != expected:
            continue
        if phrase is None or phrase in normalize(text):
            return rank
    return None


def summarize(ranks, ks):
    """hit@k for each k, plus mean reciprocal rank, from a list of ranks (None = miss)."""
    n = len(ranks)
    if n == 0:
        return {"hit_at": {k: 0.0 for k in ks}, "mrr": 0.0, "n": 0}
    return {
        "hit_at": {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks},
        "mrr": sum(1.0 / r for r in ranks if r is not None) / n,
        "n": n,
    }


def percentile(values, p):
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def build_retriever(chroma_path, collection_name, embed_model_name, top_k):
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)
    if collection.count() == 0:
        sys.exit(f"Collection '{collection_name}' at {chroma_path} is empty; index documents first.")

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device="cpu")
    index = VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=collection), embed_model=embed_model
    )
    return index.as_retriever(similarity_top_k=top_k), collection.count()


def run(questions, retrieve):
    """Run every question through retrieve(question) -> [(source, text)]."""
    results = []
    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve(q["question"])
        latency_ms = (time.perf_counter() - start) * 1000
        results.append({
            "question": q["question"],
            "expected_source": q["expected_source"],
            "doc_rank": first_hit_rank(retrieved, q["expected_source"]),
            "chunk_rank": first_hit_rank(retrieved, q["expected_source"], q.get("must_contain")),
            "top_sources": [source for source, _ in retrieved],
            "latency_ms": round(latency_ms, 2),
        })
    return results


def report(results, ks, chunk_count):
    doc = summarize([r["doc_rank"] for r in results], ks)
    chunk = summarize([r["chunk_rank"] for r in results], ks)
    latencies = [r["latency_ms"] for r in results]

    print(f"\n{len(results)} questions against {chunk_count} indexed chunks\n")
    header = "".join(f"  hit@{k:<4}" for k in ks)
    print(f"{'level':<10}{header}  MRR")
    for name, s in (("document", doc), ("chunk", chunk)):
        cells = "".join(f"  {s['hit_at'][k]:>6.1%}" for k in ks)
        print(f"{name:<10}{cells}  {s['mrr']:.3f}")
    print(f"\nretrieval latency  p50 {percentile(latencies, 50):.0f} ms   "
          f"p95 {percentile(latencies, 95):.0f} ms")

    misses = [r for r in results if r["doc_rank"] is None or r["doc_rank"] > APP_TOP_K]
    if misses:
        print(f"\nQuestions where the right act is NOT in the app's top {APP_TOP_K}:")
        for r in misses:
            got = ", ".join(r["top_sources"][:APP_TOP_K]) or "nothing"
            print(f"  - {r['question']}\n      expected {r['expected_source']}; got {got}")

    return {"document": doc, "chunk": chunk,
            "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default=os.path.join(HERE, "questions.jsonl"))
    parser.add_argument("--chroma-path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    parser.add_argument("--collection", default="lawracle_docs")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=8, help="chunks to retrieve per question")
    parser.add_argument("--json-out", help="also write per-question results to this file")
    args = parser.parse_args(argv)

    ks = sorted({k for k in (1, 2, 4, 8, args.top_k) if k <= args.top_k})
    questions = load_questions(args.questions)
    retriever, chunk_count = build_retriever(
        args.chroma_path, args.collection, args.embed_model, args.top_k
    )

    def retrieve(question):
        return [(n.node.metadata.get("source"), n.node.get_content()) for n in retriever.retrieve(question)]

    results = run(questions, retrieve)
    summary = report(results, ks, chunk_count)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
