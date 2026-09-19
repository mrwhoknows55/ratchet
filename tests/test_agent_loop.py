import json

from ratchet.agent import config as agent_config
from ratchet.agent import loop as agent_loop
from ratchet.agent.events import TurnResult


def _replies(*payloads):
    responses = iter(payloads)

    def fake_call_llm(messages, override_config=None, tools=None):
        return next(responses)

    return fake_call_llm


def _final(content="done", **extra):
    return {"status": "success", "content": content, **extra}


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "status": "success",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_run_turn_returns_a_turn_result(tmp_path):
    result = agent_loop.run_turn(_replies(_final("all set")), "hi", tmp_path)
    assert isinstance(result, TurnResult)
    assert result.text == "all set"
    assert result.status == "ok"
    assert result.steps == 1


def test_run_turn_counts_steps_and_records_tools_used(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    calls = _replies(
        _tool_call("read_files", {"path": "a.txt"}),
        _tool_call("list_files", {}, call_id="call_2"),
        _final(),
    )
    result = agent_loop.run_turn(calls, "hi", tmp_path)
    assert result.steps == 3
    assert result.tools_used == ["read_files", "list_files"]
    assert result.elapsed >= 0


def test_run_turn_accumulates_token_usage_across_steps(tmp_path):
    calls = _replies(
        _tool_call("list_files", {}) | {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        _final() | {"usage": {"prompt_tokens": 30, "completion_tokens": 5}},
    )
    result = agent_loop.run_turn(calls, "hi", tmp_path)
    assert result.prompt_tokens == 40
    assert result.completion_tokens == 7


def test_run_turn_leaves_tokens_unset_when_provider_reports_none(tmp_path):
    result = agent_loop.run_turn(_replies(_final()), "hi", tmp_path)
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


def test_run_turn_uses_the_given_system_prompt(tmp_path):
    seen = {}

    def fake_call_llm(messages, override_config=None, tools=None):
        seen["messages"] = list(messages)
        return _final()

    agent_loop.run_turn(fake_call_llm, "hi", tmp_path, system_prompt="be terse")
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}


def test_run_turn_uses_the_given_tools_schema(tmp_path):
    seen = {}
    schema = [{"type": "function", "function": {"name": "read_files"}}]

    def fake_call_llm(messages, override_config=None, tools=None):
        seen["tools"] = tools
        return _final()

    agent_loop.run_turn(fake_call_llm, "hi", tmp_path, tools_schema=schema)
    assert seen["tools"] == schema


def test_run_turn_max_steps_argument_overrides_config(tmp_path):
    calls = iter([_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)])

    def fake_call_llm(messages, override_config=None, tools=None):
        return next(calls)

    result = agent_loop.run_turn(fake_call_llm, "hi", tmp_path, max_steps=2)
    assert result.status == "max_steps"
    assert result.steps == 2


def test_run_turn_reports_error_status_when_the_llm_fails(tmp_path):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {"status": "offline", "content": "[LM Studio Offline] nope"}

    result = agent_loop.run_turn(fake_call_llm, "hi", tmp_path)
    assert result.status == "error"
    assert "Offline" in result.text


def test_run_turn_passes_depth_through_to_the_context(tmp_path, monkeypatch):
    seen = {}

    def fake_execute(name, arguments, context):
        seen["depth"] = context.depth
        seen["call_llm_fn"] = context.call_llm_fn
        return "ok", 0

    monkeypatch.setattr(agent_loop, "execute_tool_result", fake_execute)
    calls = _replies(_tool_call("list_files", {}), _final())
    agent_loop.run_turn(calls, "hi", tmp_path, depth=2)
    assert seen["depth"] == 2
    assert seen["call_llm_fn"] is not None


def test_run_agent_turn_still_returns_a_plain_string(tmp_path):
    reply = agent_loop.run_agent_turn(_replies(_final("plain")), "hi", tmp_path)
    assert reply == "plain"
    assert isinstance(reply, str)


def test_run_turn_falls_back_to_config_max_steps(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[agent]\nmax_steps = 3\n")
    monkeypatch.setattr(agent_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(agent_config, "ENV_FILE", tmp_path / ".env")

    calls = iter([_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)])

    def fake_call_llm(messages, override_config=None, tools=None):
        return next(calls)

    result = agent_loop.run_turn(fake_call_llm, "hi", tmp_path)
    assert result.steps == 3
