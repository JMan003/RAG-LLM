"""Tests for the incremental-ingestion planning logic (no ML libraries needed)."""

import hashlib

import ingest


def test_plan_sync_classifies_files():
    current = {"a.pdf": "h1", "b.pdf": "h2", "c.pdf": "h3", "d.pdf": "h4"}
    indexed = {"a.pdf": "h1", "b.pdf": "old", "c.pdf": None, "gone.pdf": "h9"}

    plan = ingest.plan_sync(current, indexed)

    assert plan.unchanged == ["a.pdf"]
    assert plan.update == ["b.pdf", "c.pdf"]  # changed, and legacy (no hash)
    assert plan.add == ["d.pdf"]
    assert plan.orphaned == ["gone.pdf"]


def test_plan_sync_never_deletes_when_data_dir_is_empty():
    plan = ingest.plan_sync({}, {"a.pdf": "h1"})
    assert plan.add == plan.update == plan.unchanged == []
    assert plan.orphaned == ["a.pdf"]


def test_list_data_files_skips_hidden_and_unsupported(tmp_path):
    for name in ["act.pdf", "notes.TXT", "readme.md", "image.png", ".upload-x.tmp", ".hidden.pdf"]:
        (tmp_path / name).write_text("x")
    (tmp_path / "subdir.pdf").mkdir()

    assert sorted(ingest.list_data_files(str(tmp_path))) == ["act.pdf", "notes.TXT", "readme.md"]


def test_list_data_files_missing_dir(tmp_path):
    assert ingest.list_data_files(str(tmp_path / "nope")) == {}


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"section 103: murder" * 1000)
    assert ingest.file_sha256(str(path), block_size=7) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_indexed_sources_prefers_known_hash(make_collection):
    collection = make_collection([
        ("1", {"source": "a.pdf"}),                     # legacy chunk
        ("2", {"source": "a.pdf", "file_hash": "h1"}),
        ("3", {"source": "b.pdf"}),
        ("4", {"other": "no source"}),
    ])
    assert ingest.indexed_sources(collection) == {"a.pdf": "h1", "b.pdf": None}


def test_delete_source_removes_only_that_source(make_collection):
    collection = make_collection([
        ("1", {"source": "a.pdf"}), ("2", {"source": "a.pdf"}), ("3", {"source": "b.pdf"}),
    ])
    assert ingest.delete_source(collection, "a.pdf") == 2
    assert ingest.chunk_counts(collection) == {"b.pdf": 1}
    assert ingest.delete_source(collection, "missing.pdf") == 0


def test_sync_index_indexes_only_new_and_changed(tmp_path, monkeypatch, make_collection):
    (tmp_path / "same.txt").write_text("unchanged text")
    (tmp_path / "new.txt").write_text("brand new text")
    (tmp_path / "empty.txt").write_text("   ")
    same_hash = ingest.file_sha256(str(tmp_path / "same.txt"))

    collection = make_collection([
        ("1", {"source": "same.txt", "file_hash": same_hash}),
        ("2", {"source": "orphan.txt", "file_hash": "x"}),
    ])

    indexed = []

    def fake_index_file(index, coll, source, path, splitter, file_hash=None):
        text = ingest.read_document_text(path)
        if not text.strip():
            return 0
        indexed.append(source)
        return 3

    monkeypatch.setattr(ingest, "index_file", fake_index_file)
    summary = ingest.sync_index(None, collection, str(tmp_path), None)

    assert indexed == ["new.txt"]
    assert summary["added"] == ["empty.txt", "new.txt"]
    assert summary["skipped"] == ["empty.txt"]
    assert summary["unchanged"] == 1
    assert summary["orphaned"] == ["orphan.txt"]
    assert summary["chunks_added"] == 3
    # Orphaned chunks are kept, not deleted.
    assert "orphan.txt" in ingest.chunk_counts(collection)
