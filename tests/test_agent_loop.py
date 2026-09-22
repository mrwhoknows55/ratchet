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


def _spawn_call(role, call_id, task="go", name="spawn_parallel_subagent"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"task": task, "role": role}),
        },
    }


def _parent_then_subagents(spawn_calls, subagent_reply):
    state = {"parent_done": False}

    def fake_call_llm(messages, override_config=None, tools=None):
        names = {t["function"]["name"] for t in (tools or [])}
        if "spawn_parallel_subagent" in names:
            if state["parent_done"]:
                return _final("parent done")
            state["parent_done"] = True
            return {"status": "success", "content": "", "tool_calls": spawn_calls}
        return subagent_reply()

    return fake_call_llm


def test_parallel_spawns_of_a_read_only_role_run_concurrently(tmp_path):
    import threading

    barrier = threading.Barrier(2, timeout=5)

    def subagent_reply():
        barrier.wait()
        return _final("sub done")

    calls = [_spawn_call("researcher", "c1"), _spawn_call("researcher", "c2")]
    messages = [{"role": "system", "content": "parent"}]
    result = agent_loop.run_turn(
        _parent_then_subagents(calls, subagent_reply), "hi", tmp_path, messages=messages
    )
    assert result.status == "ok"
    assert not barrier.broken


def _concurrency_probe():
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def subagent_reply():
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return _final("sub done")

    return state, subagent_reply


def test_plain_spawn_calls_never_run_in_parallel(tmp_path):
    state, subagent_reply = _concurrency_probe()
    calls = [
        _spawn_call("researcher", "c1", name="spawn_subagent"),
        _spawn_call("researcher", "c2", name="spawn_subagent"),
    ]
    agent_loop.run_turn(
        _parent_then_subagents(calls, subagent_reply),
        "hi",
        tmp_path,
        messages=[{"role": "system", "content": "parent"}],
    )
    assert state["peak"] == 1


def test_ordinary_tool_calls_never_run_in_parallel(tmp_path, monkeypatch):
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def fake_execute(name, arguments, context):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return "ok", 0

    monkeypatch.setattr(agent_loop, "execute_tool_result", fake_execute)
    both = {
        "status": "success",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            },
        ],
    }
    agent_loop.run_turn(_replies(both, _final()), "hi", tmp_path)
    assert state["peak"] == 1


def test_parallel_tool_messages_keep_call_order(tmp_path):
    import time

    order = iter([0.06, 0.0])

    def subagent_reply():
        time.sleep(next(order))
        return _final("sub done")

    calls = [_spawn_call("researcher", "c1"), _spawn_call("researcher", "c2")]
    messages = [{"role": "system", "content": "parent"}]
    agent_loop.run_turn(
        _parent_then_subagents(calls, subagent_reply), "hi", tmp_path, messages=messages
    )
    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"]


def test_parallel_lanes_flush_their_events_without_interleaving(tmp_path):
    events = []
    calls = [_spawn_call("researcher", "c1"), _spawn_call("researcher", "c2")]
    agent_loop.run_turn(
        _parent_then_subagents(calls, lambda: _final("sub done")),
        "hi",
        tmp_path,
        on_event=events.append,
        messages=[{"role": "system", "content": "parent"}],
    )
    lanes = [e.lane for e in events if e.depth == 1]
    assert lanes == sorted(lanes)
    assert set(lanes) == {1, 2}


def test_max_parallel_caps_the_worker_count(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[subagent]\nmax_steps = 8\nmax_parallel = 2\n")
    monkeypatch.setattr(agent_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(agent_config, "ENV_FILE", tmp_path / ".env")
    state, subagent_reply = _concurrency_probe()
    calls = [_spawn_call("researcher", f"c{i}") for i in range(4)]
    agent_loop.run_turn(
        _parent_then_subagents(calls, subagent_reply),
        "hi",
        tmp_path,
        messages=[{"role": "system", "content": "parent"}],
    )
    assert state["peak"] == 2
