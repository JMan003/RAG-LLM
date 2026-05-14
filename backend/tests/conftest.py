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
