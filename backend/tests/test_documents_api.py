"""Tests for the document upload / list / delete endpoints (models mocked)."""

import hashlib
import io
from unittest.mock import MagicMock

import pytest

import app as app_module
import ingest


@pytest.fixture
def ready_app(mocker, tmp_path, make_collection):
    """App state as if initialisation finished, with a temp data dir."""
    collection = make_collection([("1", {"source": "old act.pdf", "file_hash": "h0"})])
    mocker.patch.object(app_module, 'query_engine', MagicMock())
    mocker.patch.object(app_module, 'init_error', None)
    mocker.patch.object(app_module, 'chroma_collection', collection)
    mocker.patch.object(app_module, 'rag_index', MagicMock())
    mocker.patch.object(app_module, 'text_splitter', MagicMock())
    mocker.patch.object(app_module, 'DATA_DIR', str(tmp_path))
    index_file = mocker.patch.object(ingest, 'index_file', return_value=5)
    return {"collection": collection, "index_file": index_file, "data_dir": tmp_path}


def upload(client, name, content):
    return client.post(
        '/documents',
        data={'file': (io.BytesIO(content), name)},
        content_type='multipart/form-data',
    )


def test_upload_rejected_while_initializing(client, mocker):
    mocker.patch.object(app_module, 'query_engine', None)
    assert upload(client, 'a.txt', b'text').status_code == 503


def test_upload_requires_file(client, ready_app):
    response = client.post('/documents', data={}, content_type='multipart/form-data')
    assert response.status_code == 400


def test_upload_rejects_unsupported_type(client, ready_app):
    response = upload(client, 'payload.exe', b'MZ')
    assert response.status_code == 400
    ready_app["index_file"].assert_not_called()


def test_upload_rejects_file_without_text(client, ready_app):
    response = upload(client, 'blank.txt', b'   \n  ')
    assert response.status_code == 422
    assert list(ready_app["data_dir"].iterdir()) == []


def test_upload_adds_document(client, ready_app):
    response = upload(client, 'new act.txt', b'Section 1. Short title.')
    assert response.status_code == 201
    assert response.json == {"source": "new_act.txt", "status": "added", "chunks": 5}
    assert (ready_app["data_dir"] / "new_act.txt").read_bytes() == b'Section 1. Short title.'
    # No temp files left behind.
    assert [p.name for p in ready_app["data_dir"].iterdir()] == ["new_act.txt"]


def test_upload_same_content_is_unchanged(client, ready_app):
    content = b'Section 2. Definitions.'
    file_hash = hashlib.sha256(content).hexdigest()
    ready_app["collection"].chunks.append(("2", {"source": "defs.txt", "file_hash": file_hash}))

    response = upload(client, 'defs.txt', content)
    assert response.status_code == 200
    assert response.json["status"] == "unchanged"
    ready_app["index_file"].assert_not_called()


def test_list_documents(client, ready_app):
    response = client.get('/documents')
    assert response.status_code == 200
    assert response.json == {
        "documents": [{"source": "old act.pdf", "chunks": 1}],
        "total_chunks": 1,
    }


def test_delete_document_with_spaces_in_name(client, ready_app):
    (ready_app["data_dir"] / "old act.pdf").write_bytes(b'%PDF')
    response = client.delete('/documents/old%20act.pdf')
    assert response.status_code == 200
    assert response.json == {"source": "old act.pdf", "removed_chunks": 1}
    assert not (ready_app["data_dir"] / "old act.pdf").exists()


def test_delete_unknown_document(client, ready_app):
    assert client.delete('/documents/missing.pdf').status_code == 404


def test_delete_rejects_path_traversal(client, ready_app):
    assert client.delete('/documents/..%2Fapp.py').status_code in (400, 404)
    assert client.delete('/documents/.hidden.pdf').status_code == 400
