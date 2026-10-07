import json

import pytest

from ratchet.agent import config as agent_config
from ratchet.agent import loop as agent_loop
from ratchet.agent.events import TurnResult
from ratchet.agent.plan import EXECUTE_PLAN_PROMPT, create_plan


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


def test_invalid_response_is_retried_once_without_polluting_history(tmp_path):
    seen = []
    history = []

    def fake_llm(messages, override_config=None, tools=None):
        seen.append(list(messages))
        if len(seen) == 1:
            return {"status": "invalid_response", "content": "[Error] malformed model output",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        assert any("structured tool_calls" in m["content"] for m in messages)
        return _final("Recovered", usage={"prompt_tokens": 110, "completion_tokens": 10})

    result = agent_loop.run_turn(fake_llm, "inspect", tmp_path, messages=history)
    assert result.status == "ok"
    assert result.text == "Recovered"
    assert len(seen) == 2
    assert result.prompt_tokens == 210
    assert result.completion_tokens == 30
    assert history == [{"role": "user", "content": "inspect"},
                       {"role": "assistant", "content": "Recovered"}]


def test_repeated_invalid_responses_stop_after_one_retry(tmp_path):
    seen = []

    def fake_llm(messages, override_config=None, tools=None):
        seen.append(list(messages))
        return {"status": "invalid_response", "content": "[Error] malformed model output"}

    result = agent_loop.run_turn(fake_llm, "inspect", tmp_path)
    assert result.status == "error"
    assert len(seen) == 2
    assert result.text == "[Error] malformed model output"


def test_malformed_tool_arguments_are_reported_back_to_the_model(tmp_path):
    bad_call = _tool_call("read_files", {})
    bad_call["tool_calls"][0]["function"]["arguments"] = '{"path": "a.txt"'
    seen = []

    def fake_call_llm(messages, override_config=None, tools=None):
        seen.append(list(messages))
        return bad_call if len(seen) == 1 else _final("recovered")

    result = agent_loop.run_turn(fake_call_llm, "hi", tmp_path)
    assert result.text == "recovered"
    tool_message = seen[1][-1]
    assert tool_message["role"] == "tool"
    assert "invalid JSON arguments" in tool_message["content"]


def test_plan_mode_offers_read_only_tools_and_tells_the_model(tmp_path):
    seen = {}

    def fake_call_llm(messages, override_config=None, tools=None):
        seen["tools"] = {schema["function"]["name"] for schema in tools}
        seen["user"] = messages[-1]["content"]
        return _final("planned")

    agent_loop.run_turn(fake_call_llm, "design it", tmp_path, plan_mode=True)
    assert "create_plan" in seen["tools"]
    assert "write_files" not in seen["tools"]
    assert "design it" in seen["user"]
    assert "plan mode" in seen["user"].lower()


def test_plan_mode_blocks_a_mutating_call_the_model_sends_anyway(tmp_path):
    fake = _replies(_tool_call("write_files", {"path": "a.txt", "content": "x"}), _final())
    events = []
    agent_loop.run_turn(fake, "hi", tmp_path, on_event=events.append, plan_mode=True)
    assert not (tmp_path / "a.txt").exists()
    done = [e for e in events if e.phase == "tool_done"][0]
    assert done.exit_code == 1
    assert "plan mode" in done.output.lower()


@pytest.mark.parametrize("prompt", [EXECUTE_PLAN_PROMPT, "execute"])
def test_execute_plan_tracks_a_real_file_change_and_verification(tmp_path, prompt):
    create_plan(tmp_path, "Greeting", ["Write greeting"])
    messages = []
    calls = _replies(
        _tool_call("get_plan", {}, "get"),
        _tool_call("update_plan_step", {"step_id": 1, "status": "in_progress"}, "start"),
        _tool_call("write_files", {"path": "hello.txt", "content": "Hello!"}, "write"),
        _tool_call("read_files", {"path": "hello.txt"}, "verify"),
        _tool_call("update_plan_step", {
            "step_id": 1, "status": "completed", "notes": "Read back hello.txt",
        }, "finish"),
        _final("Executed and verified"),
    )

    def fake_llm(history, override_config=None, tools=None):
        request = "\n".join(m["content"] for m in history if m["role"] == "system")
        assert "get_plan" in request
        assert "update_plan_step" in request
        assert "in_progress" in request
        assert "verification" in request
        assert next(m["content"] for m in reversed(history) if m["role"] == "user") == prompt
        return calls(history, override_config, tools=tools)

    result = agent_loop.run_turn(fake_llm, prompt, tmp_path, messages=messages)
    assert result.status == "ok"
    assert (tmp_path / "hello.txt").read_text() == "Hello!"
    outputs = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert json.loads(outputs["start"])["plan"]["steps"][0]["status"] == "in_progress"
    assert "Hello!" in outputs["verify"]
    assert json.loads(outputs["finish"])["progress"]["percent"] == 100.0
    assert "- [x] Write greeting (completed)" in (tmp_path / "PLAN.md").read_text()


def test_execution_after_planning_receives_current_mode_and_tracking_instructions(tmp_path):
    messages = []
    agent_loop.run_turn(_replies(
        _tool_call("create_plan", {"title": "Greeting", "steps": ["Write greeting"]}),
        _final("Review the plan"),
    ), "plan greeting", tmp_path, messages=messages, plan_mode=True)

    def fake_llm(history, override_config=None, tools=None):
        request = history[0]["content"]
        assert request.startswith("[Build mode]")
        assert "[Plan mode] Read-only" in history[1]["content"]
        assert "get_plan" in request
        assert "update_plan_step" in request
        assert "failed" in request
        assert "delegated" in request
        assert history[-1] == {"role": "user", "content": "yes execute"}
        return _final("Ready")

    agent_loop.run_turn(fake_llm, "yes execute", tmp_path, messages=messages)
    assert not any(m["role"] == "system" for m in messages)
    assert json.loads((tmp_path / "PLAN.json").read_text())["steps"][0]["status"] == "pending"


@pytest.mark.parametrize("state", ["missing", "corrupt", "restricted", "planning", "nested"])
def test_plan_execution_reminder_respects_plan_state_and_agent_permissions(tmp_path, state):
    if state == "corrupt":
        (tmp_path / "PLAN.json").write_text("{broken")
    elif state != "missing":
        create_plan(tmp_path, "Greeting", ["Write greeting"])
    kwargs = {}
    if state == "restricted":
        kwargs["tools_schema"] = [{"type": "function", "function": {"name": "read_files"}}]
    if state == "planning":
        kwargs["plan_mode"] = True
    if state == "nested":
        kwargs["depth"] = 1

    def fake_llm(history, override_config=None, tools=None):
        request = history[-1]["content"]
        assert not any("[Build mode]" in m["content"] for m in history)
        if state == "planning":
            assert "[Plan mode]" in request
        else:
            assert request == "inspect greeting"
        return _final("Inspected")

    agent_loop.run_turn(fake_llm, "inspect greeting", tmp_path, **kwargs)


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
    assert result.status == "max_steps"
    assert result.steps == 3
    assert "exceeded" in result.text.lower()


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


def test_plan_mode_delegates_plan_creation_to_a_planner(tmp_path):
    events = []
    calls = _replies(
        _tool_call("spawn_subagent", {"task": "Design greeting", "role": "planner"}),
        _tool_call("read_files", {"path": "hello.txt"}),
        _tool_call("create_plan", {"title": "Greeting", "steps": ["Improve greeting"]}),
        _final("Planner summary"),
        _final("Review the plan"),
    )
    (tmp_path / "hello.txt").write_text("Hello!")
    result = agent_loop.run_turn(calls, "plan greeting", tmp_path,
                                 plan_mode=True, on_event=events.append)
    assert result.status == "ok"
    assert (tmp_path / "hello.txt").read_text() == "Hello!"
    assert json.loads((tmp_path / "PLAN.json").read_text())["title"] == "Greeting"
    nested = [e for e in events if e.depth == 1 and e.phase == "tool_done"]
    assert [e.name for e in nested] == ["read_files", "create_plan"]
    assert all(e.agent == "planner" and e.exit_code == 0 for e in nested)


def test_planning_parallel_research_inherits_constraints_and_role_permissions(tmp_path):
    import threading

    barrier = threading.Barrier(2, timeout=5)
    events = []
    names_seen = []
    lock = threading.Lock()
    parent_done = False

    def fake_llm(messages, override_config=None, tools=None):
        nonlocal parent_done
        names = {s["function"]["name"] for s in tools}
        if "spawn_subagent" in names:
            if parent_done:
                return _final("Reviewed research")
            parent_done = True
            return {"status": "success", "content": "", "tool_calls": [
                _spawn_call("researcher", "one"), _spawn_call("researcher", "two"),
            ]}
        if messages[-1]["role"] == "tool":
            return _final("Blocked unexpected edit")
        with lock:
            names_seen.append(names)
        barrier.wait()
        return _tool_call("create_plan", {"title": "Bad", "steps": ["Bad"]})

    result = agent_loop.run_turn(fake_llm, "inspect", tmp_path,
                                 plan_mode=True, on_event=events.append)
    assert result.status == "ok"
    assert len(names_seen) == 2
    assert all("read_files" in names and "create_plan" not in names for names in names_seen)
    blocked = [e for e in events if e.depth == 1 and e.phase == "tool_done"]
    assert len(blocked) == 2
    assert all(e.exit_code == 1 and "blocked" in e.output.lower() for e in blocked)
    assert {e.lane for e in blocked} == {1, 2}
    assert not (tmp_path / "PLAN.json").exists()
