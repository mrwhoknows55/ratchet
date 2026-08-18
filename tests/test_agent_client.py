import httpx
import pytest

from ratchet.agent import client as agent_client
from ratchet.agent import config as agent_config


@pytest.fixture(autouse=True)
def _isolate_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_config, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(agent_config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_call_llm_returns_success_content(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer not-needed"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello there"}}],
            },
        )

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result == {
        "content": "hello there",
        "model": "qwen/qwen3-4b-2507",
        "status": "success",
    }


def test_call_llm_handles_none_content_as_empty_string(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None}}]},
        )

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["content"] == ""
    assert result["status"] == "success"


def test_call_llm_reports_offline_on_connect_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["status"] == "offline"
    assert "http://localhost:1234/v1" in result["content"]
    assert "error" in result


def test_call_llm_reports_error_on_http_status_error(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["status"] == "error"
    assert "error" in result


def test_call_llm_reports_error_on_unexpected_exception(monkeypatch):
    def handler(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["status"] == "error"
    assert "boom" in result["error"]


def test_call_llm_override_config_takes_precedence(monkeypatch):
    import json

    seen = {}

    def handler(request):
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm(
        [{"role": "user", "content": "hi"}],
        override_config={"model": {"name": "override-model"}},
    )

    assert seen["model"] == "override-model"
    assert result["model"] == "override-model"


def test_call_llm_includes_tools_in_request_when_provided(monkeypatch):
    import json

    seen = {}
    tools = [{"type": "function", "function": {"name": "list_files"}}]

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    agent_client.call_llm([{"role": "user", "content": "hi"}], tools=tools)

    assert seen["body"]["tools"] == tools


def test_call_llm_omits_tools_key_when_not_provided(monkeypatch):
    import json

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert "tools" not in seen["body"]


def test_call_llm_returns_tool_calls_when_present(monkeypatch):
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}
    ]

    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]},
        )

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["tool_calls"] == tool_calls
    assert result["content"] == ""


def test_call_llm_env_vars_take_precedence_over_config(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host:9999/v1")

    def handler(request):
        assert request.url.host == "env-host"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr(agent_client, "_transport", _mock_transport(handler))

    result = agent_client.call_llm([{"role": "user", "content": "hi"}])

    assert result["model"] == "env-model"
