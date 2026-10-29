"""Tests for the retrieval-evaluation scoring (no index or models needed)."""

import os

from evaluation import eval_retrieval as ev


def test_first_hit_rank_document_and_chunk_level():
    retrieved = [
        ("IT Act.pdf", "Section 66C. Punishment for\n identity   theft."),
        ("RTI Act.pdf", "within thirty days"),
        ("IT Act.pdf", "cyber terrorism"),
    ]
    assert ev.first_hit_rank(retrieved, "IT Act.pdf") == 1
    # Phrase match ignores case and PDF line breaks.
    assert ev.first_hit_rank(retrieved, "IT Act.pdf", "punishment for IDENTITY theft") == 1
    assert ev.first_hit_rank(retrieved, "IT Act.pdf", "cyber terrorism") == 3
    assert ev.first_hit_rank(retrieved, "POCSO Act.pdf") is None
    assert ev.first_hit_rank(retrieved, "RTI Act.pdf", "not there") is None


def test_normalize_unifies_curly_quotes():
    assert ev.normalize("“consumer”  means") == '"consumer" means'


def test_summarize_hit_rates_and_mrr():
    s = ev.summarize([1, 2, None, 4], ks=[1, 2, 4])
    assert s["hit_at"] == {1: 0.25, 2: 0.5, 4: 0.75}
    assert abs(s["mrr"] - (1 + 0.5 + 0.25) / 4) < 1e-9
    assert ev.summarize([], ks=[1])["mrr"] == 0.0


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert ev.percentile(values, 50) == 10
    assert ev.percentile(values, 95) == 19
    assert ev.percentile([7], 95) == 7
    assert ev.percentile([], 50) == 0.0


def test_run_records_ranks_and_latency():
    questions = [{"question": "q1", "expected_source": "a.pdf", "must_contain": "needle"}]
    results = ev.run(questions, lambda q: [("b.pdf", "x"), ("a.pdf", "a NEEDLE here")])
    assert results[0]["doc_rank"] == 2
    assert results[0]["chunk_rank"] == 2
    assert results[0]["top_sources"] == ["b.pdf", "a.pdf"]
    assert results[0]["latency_ms"] >= 0


def test_question_set_is_well_formed():
    path = os.path.join(os.path.dirname(ev.__file__), "questions.jsonl")
    rows = ev.load_questions(path)
    assert len(rows) >= 30
    for row in rows:
        assert set(row) == {"question", "expected_source", "must_contain"}
        assert row["expected_source"].endswith(".pdf")
        assert row["must_contain"].strip()
    # Questions are unique.
    assert len({r["question"] for r in rows}) == len(rows)


def test_report_lists_questions_missed_at_app_top_k(capsys):
    results = [
        {"question": "hit", "expected_source": "a.pdf", "doc_rank": 1, "chunk_rank": 1,
         "top_sources": ["a.pdf", "b.pdf"], "latency_ms": 10.0},
        {"question": "late", "expected_source": "a.pdf", "doc_rank": 3, "chunk_rank": None,
         "top_sources": ["b.pdf", "c.pdf", "a.pdf"], "latency_ms": 30.0},
    ]
    summary = ev.report(results, ks=[1, 2, 4], chunk_count=100)
    out = capsys.readouterr().out
    assert summary["document"]["hit_at"][4] == 1.0
    assert summary["chunk"]["hit_at"][4] == 0.5
    assert "late" in out and "hit\n" not in out
