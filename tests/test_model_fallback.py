import json

import httpx
import pytest

from ratchet import cli
from ratchet.agent import client as agent_client
from ratchet.agent.loop import run_turn
from ratchet.tui import main as tui_main


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(agent_client, "_context_windows", {})
    cfg = {
        "model": {"name": "local-model", "base_url": "http://localhost:1234/v1",
                  "api_key": "local-key", "context_window": 4096},
        "models": {"gptoss": {"name": "openai/gpt-oss-120b",
                              "base_url": "https://openrouter.ai/api/v1",
                              "api_key": "fallback-key", "fallback": True}},
    }
    monkeypatch.setattr(agent_client, "load_config", lambda: cfg)
    for name in ("OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return cfg


@pytest.mark.parametrize("failure", ["400", "offline", "timeout", "malformed"])
def test_failed_primary_uses_enabled_fallback_with_the_same_messages_and_tools(
    config, monkeypatch, failure,
):
    requests = []
    messages = [{"role": "user", "content": "design safe division"}]
    tools = [{"type": "function", "function": {"name": "create_plan"}}]

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{
                "id": "openai/gpt-oss-120b", "context_length": 131072,
            }]})
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        if request.url.host == "localhost":
            if failure == "offline":
                raise httpx.ConnectError("offline", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            if failure == "malformed":
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": "analysisPrivate reasoning without a final answer.",
                }}]})
            return httpx.Response(400, json={"error": {"message": "unsupported model"}})
        assert request.headers["authorization"] == "Bearer fallback-key"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Plan ready"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm(messages, tools=tools)
    assert result["status"] == "success"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["content"] == "Plan ready"
    assert result["fallback_from"] == "local-model"
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert result["context_window"] == 131072
    assert [host for host, _ in requests] == ["localhost", "openrouter.ai"]
    assert all(body["messages"] == messages and body["tools"] == tools for _, body in requests)
    assert requests[1][1]["model"] == "openai/gpt-oss-120b"


@pytest.mark.parametrize("enabled", [False, None, "true"])
def test_fallback_requires_an_explicit_true_boolean(config, monkeypatch, enabled):
    config["models"]["gptoss"]["fallback"] = enabled
    requests = []

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(400)

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([])
    assert result["status"] == "error"
    assert requests == ["localhost"]


def test_successful_primary_does_not_use_fallback(config, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Local success"}}]})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([])
    assert result["model"] == "local-model"
    assert "fallback_from" not in result
    assert requests == ["localhost"]


def test_primary_environment_overrides_do_not_redirect_fallback(config, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://primary-host:1234/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-primary")
    monkeypatch.setenv("OPENAI_API_KEY", "primary-key")
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body["model"]))
        if request.url.host == "primary-host":
            assert request.headers["authorization"] == "Bearer primary-key"
            return httpx.Response(400)
        assert request.headers["authorization"] == "Bearer fallback-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered"}}]})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([])
    assert result["status"] == "success"
    assert result["fallback_from"] == "env-primary"
    assert requests == [("primary-host", "env-primary"), ("openrouter.ai", "openai/gpt-oss-120b")]


def test_failed_fallback_stops_after_one_attempt_and_reports_both_errors(config, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(400 if request.url.host == "localhost" else 503)

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([])
    assert result["status"] == "error"
    assert requests == ["localhost", "openrouter.ai"]
    assert "400" in result["content"]
    assert "503" in result["content"]


def test_fallback_never_retries_itself(config, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(503)

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    result = agent_client.call_llm([], {"model": config["models"]["gptoss"]})
    assert result["status"] == "error"
    assert requests == ["openrouter.ai"]


def test_loop_emits_a_fallback_notice(config, monkeypatch, tmp_path):
    def handler(request):
        if request.url.host == "localhost":
            return httpx.Response(400)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Plan ready"}}]})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    events = []
    result = run_turn(agent_client.call_llm, "design", tmp_path, on_event=events.append)
    assert result.status == "ok"
    notices = [event for event in events if event.phase == "fallback"]
    assert len(notices) == 1
    assert notices[0].name == "openai/gpt-oss-120b"
    assert notices[0].arguments == {"from": "local-model"}


def test_cli_shows_the_actual_fallback_model(config, monkeypatch, tmp_path, capsys):
    def handler(request):
        if request.url.host == "localhost":
            return httpx.Response(400)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Plan ready"}}]})

    monkeypatch.setattr(agent_client, "_transport", httpx.MockTransport(handler))
    cli.run_cli("design", sandbox_root=tmp_path, session_path=tmp_path / "session.json",
                plan_mode=True)
    output = capsys.readouterr().out
    assert "Fallback" in output
    assert "openai/gpt-oss-120b" in output
    assert "Plan ready" in output


async def test_tui_shows_and_logs_fallback_notice(monkeypatch, tmp_path):
    monkeypatch.setattr(tui_main, "call_llm", lambda *args, **kwargs: {
        "status": "success", "content": "Plan ready", "model": "openai/gpt-oss-120b",
        "fallback_from": "local-model",
    })
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        app.query_one("#message_input", tui_main.PromptInput).text = "design"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        transcript = "\n".join(
            line.text for line in app.query_one("#messages", tui_main.RichLog).lines
        )
        assert "Fallback: local-model → openai/gpt-oss-120b" in transcript
        assert "Fallback: local-model → openai/gpt-oss-120b" in app.log_path.read_text()
