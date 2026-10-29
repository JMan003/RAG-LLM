import pytest
from app import app

@pytest.fixture
def client():
    # Configure the app for testing
    app.config.update({
        "TESTING": True,
    })

    # Create a test client
    with app.test_client() as client:
        # Establish an application context before running the tests.
        with app.app_context():
            yield client


class FakeCollection:
    """Minimal stand-in for a Chroma collection: ids + metadatas."""

    def __init__(self, chunks):
        # chunks: list of (id, metadata)
        self.chunks = list(chunks)

    def get(self, where=None, include=None):
        rows = [
            (cid, md) for cid, md in self.chunks
            if where is None or all(md.get(k) == v for k, v in where.items())
        ]
        return {"ids": [cid for cid, _ in rows], "metadatas": [md for _, md in rows]}

    def delete(self, ids):
        drop = set(ids)
        self.chunks = [(cid, md) for cid, md in self.chunks if cid not in drop]


@pytest.fixture
def make_collection():
    return FakeCollection
