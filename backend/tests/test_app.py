import pytest
from unittest.mock import MagicMock
from app import truncate_text, format_leap_response
import app as app_module

def test_truncate_text():
    # Generate exactly 300 words
    text = "word " * 300
    truncated = truncate_text(text, 200)
    assert len(truncated.split()) == 200
    assert truncated.endswith("...")

    short_text = "short text"
    assert truncate_text(short_text, 200) == short_text

def test_format_leap_response_accident():
    res = format_leap_response("I had a car accident", "raw context from LLM")
    assert "motor vehicle accident" in res
    assert "Explanation of Law" in res
    assert "raw context from LLM" in res

def test_format_leap_response_property():
    res = format_leap_response("dispute with my tenant", "raw context from LLM")
    assert "property dispute" in res

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'

def test_log_endpoint(client):
    response = client.post('/log', json={'message': 'test error'})
    assert response.status_code == 200
    assert response.json['status'] == 'logged'

def test_query_endpoint_uninitialized(client):
    # Testing when query_engine is None
    response = client.post('/query', json={'query': 'test'})
    assert response.status_code == 503
    assert 'Pipeline initializing' in response.json['error']

def test_query_endpoint_initialized(client, mocker):
    # Mock the query engine so it doesn't try to use the real LLM
    mock_engine = MagicMock()
    mock_engine.query.return_value = "Mocked LLM response"
    mocker.patch.object(app_module, 'query_engine', mock_engine)

    response = client.post('/query', json={'query': 'car accident'})
    
    assert response.status_code == 200
    assert "Mocked LLM response" in response.json['response']
    assert "motor vehicle accident" in response.json['response']
