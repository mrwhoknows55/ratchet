import json

import httpx
import pytest

from ratchet.agent import client as agent_client


@pytest.fixture
def config(monkeypatch):
    cfg = {"model": {"name": "openai/gpt-oss-120b",
                     "base_url": "https://openrouter.ai/api/v1", "api_key": "test-key"}}
    monkeypatch.setattr(agent_client, "load_config", lambda: cfg)
    monkeypatch.setattr(agent_client, "_context_windows", {}, raising=False)
    for name in ("OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return cfg


def test_openrouter_model_metadata_supplies_and_caches_the_context_window(config, monkeypatch):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            assert request.url.path == "/api/v1/models"
            assert request.headers["authorization"] == "Bearer test-key"
            return httpx.Response(200, json={"data": [{
                "id": "openai/gpt-oss-120b", "context_length": 131072,
                "top_provider": {"context_length": 131072, "max_completion_tokens": 117964},
            }]})
        return httpx.Response(200, json={
            "model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "Done"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        })

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    for _ in range(2):
        result = agent_client.call_llm([])
        assert result["context_window"] == 131072
        assert result["usage"] == {"prompt_tokens": 1000, "completion_tokens": 100}
    assert requests == [("POST", "/api/v1/chat/completions"), ("GET", "/api/v1/models"),
                        ("POST", "/api/v1/chat/completions")]


def test_context_lookup_uses_the_model_reported_by_the_response(config, monkeypatch):
    config["model"]["name"] = "model-alias"

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{
                "id": "openai/gpt-oss-120b", "context_length": 131072,
            }]})
        assert json.loads(request.content)["model"] == "model-alias"
        return httpx.Response(200, json={
            "model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "Done"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        })

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([])
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["context_window"] == 131072


def test_provider_limit_takes_precedence_over_a_larger_model_limit(config, monkeypatch):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{
                "id": "openai/gpt-oss-120b", "context_length": 131072,
                "top_provider": {"context_length": 65536},
            }]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}}],
                                       "usage": {"prompt_tokens": 1000}})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    assert agent_client.call_llm([])["context_window"] == 65536


@pytest.mark.parametrize("failure", ["http", "json", "invalid_limit"])
def test_missing_metadata_is_nonfatal_and_retried_on_the_next_request(config, monkeypatch, failure):
    reads = 0

    def handler(request):
        nonlocal reads
        if request.method == "GET":
            reads += 1
            if reads == 1:
                if failure == "http":
                    return httpx.Response(503)
                if failure == "json":
                    return httpx.Response(200, content="broken JSON")
                return httpx.Response(200, json={"data": [{
                    "id": "openai/gpt-oss-120b", "context_length": True,
                }]})
            return httpx.Response(200, json={"data": [{
                "id": "openai/gpt-oss-120b", "context_length": 131072,
            }]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}}],
                                       "usage": {"prompt_tokens": 1000}})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    first = agent_client.call_llm([])
    assert first["status"] == "success"
    assert "context_window" not in first
    assert agent_client.call_llm([])["context_window"] == 131072
    assert reads == 2


def test_local_loaded_window_is_refreshed_instead_of_using_the_architecture_maximum(
    config, monkeypatch,
):
    config["model"]["base_url"] = "http://localhost:1234/v1"
    windows = iter([4096, 8192])

    def handler(request):
        if request.method == "GET":
            assert request.url.path == "/api/v1/models"
            return httpx.Response(200, json={"models": [{
                "key": "openai/gpt-oss-120b", "max_context_length": 131072,
                "loaded_instances": [{"id": "openai/gpt-oss-120b", "config": {
                    "context_length": next(windows),
                }}],
            }]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}}],
                                       "usage": {"prompt_tokens": 1000}})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    assert agent_client.call_llm([])["context_window"] == 4096
    assert agent_client.call_llm([])["context_window"] == 8192
