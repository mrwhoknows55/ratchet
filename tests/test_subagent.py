import pytest

from ratchet.agent import config as agent_config
from ratchet.agent import subagent
from ratchet.agent.context import AgentContext
from ratchet.agent.events import TurnEvent
from ratchet.agent.tools import TOOL_SCHEMAS

ALL_TOOLS = {schema["function"]["name"] for schema in TOOL_SCHEMAS} - {
    "spawn_subagent",
    "spawn_parallel_subagent",
}
MUTATING = {
    "write_files",
    "replace_in_file",
    "append_file",
    "delete_file",
    "copy_file",
    "move_file",
    "rollback_file",
}


def test_roles_cover_the_four_named_subagents():
    assert set(subagent.ROLE_TOOLS) == {"researcher", "coder", "tester", "generalist"}


def test_every_role_only_names_real_tools():
    for role, names in subagent.ROLE_TOOLS.items():
        assert set(names) <= ALL_TOOLS, role


def test_researcher_cannot_mutate_or_run_commands():
    tools = set(subagent.ROLE_TOOLS["researcher"])
    assert not tools & MUTATING
    assert "run_command" not in tools
    assert {"read_files", "search_files", "search_web"} <= tools


def test_coder_can_mutate_but_not_run_commands_or_browse():
    tools = set(subagent.ROLE_TOOLS["coder"])
    assert MUTATING <= tools
    assert "run_command" not in tools
    assert "search_web" not in tools
    assert "read_files" in tools


def test_tester_can_run_commands_but_not_mutate():
    tools = set(subagent.ROLE_TOOLS["tester"])
    assert "run_command" in tools
    assert "check_command" in tools
    assert not tools & MUTATING


def test_generalist_gets_every_tool():
    assert set(subagent.ROLE_TOOLS["generalist"]) == ALL_TOOLS


def test_schema_for_role_filters_to_that_role():
    names = {s["function"]["name"] for s in subagent.schema_for_role("researcher")}
    assert names == set(subagent.ROLE_TOOLS["researcher"])


def test_schema_for_role_never_offers_spawn_subagent():
    for role in subagent.ROLE_TOOLS:
        names = {s["function"]["name"] for s in subagent.schema_for_role(role)}
        assert "spawn_subagent" not in names


def test_schema_for_role_falls_back_to_generalist_for_an_unknown_role():
    names = {s["function"]["name"] for s in subagent.schema_for_role("wizard")}
    assert names == ALL_TOOLS


@pytest.mark.parametrize("role", ["researcher", "coder", "tester", "generalist"])
def test_every_role_has_a_prompt_that_demands_a_summary(role):
    prompt = subagent.prompt_for_role(role)
    assert prompt.strip()
    assert "summary" in prompt.lower()


def test_prompt_for_role_falls_back_to_generalist():
    assert subagent.prompt_for_role("wizard") == subagent.prompt_for_role("generalist")


def test_role_prompts_stay_small():
    for role in subagent.ROLE_TOOLS:
        assert len(subagent.prompt_for_role(role)) < 1400, role



def _ctx(tmp_path, call_llm_fn, depth=0, on_event=None, override_config=None):
    return AgentContext(
        sandbox_root=tmp_path,
        call_llm_fn=call_llm_fn,
        override_config=override_config,
        on_event=on_event,
        depth=depth,
    )


def _final(content, **extra):
    return {"status": "success", "content": content, **extra}


def _once(payload):
    def fake_call_llm(messages, override_config=None, tools=None):
        return payload

    return fake_call_llm


def test_spawn_subagent_schema_is_registered_and_requires_a_task():
    schema = next(
        s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "spawn_subagent"
    )
    assert schema["parameters"]["required"] == ["task"]
    assert set(schema["parameters"]["properties"]) == {"task", "role", "context", "max_steps"}
    assert set(schema["parameters"]["properties"]["role"]["enum"]) == set(subagent.ROLE_TOOLS)


def test_spawn_subagent_returns_the_summary_and_a_metadata_line(tmp_path):
    payload = _final("found it in loop.py", usage={"prompt_tokens": 40, "completion_tokens": 8})
    result = subagent.spawn_subagent(_ctx(tmp_path, _once(payload)), "find the loop", "researcher")
    assert result["exit_code"] == 0
    assert result["stdout"].startswith("found it in loop.py")
    meta = result["stdout"].splitlines()[-1]
    assert "researcher" in meta
    assert "1/8 steps" in meta
    assert "48 tok" in meta


def test_spawn_subagent_omits_tokens_when_the_provider_reports_none(tmp_path):
    result = subagent.spawn_subagent(_ctx(tmp_path, _once(_final("done"))), "x", "researcher")
    assert "tok" not in result["stdout"].splitlines()[-1]


def test_spawn_subagent_runs_an_isolated_history_with_the_role_prompt_and_schema(tmp_path):
    seen = {}

    def fake_call_llm(messages, override_config=None, tools=None):
        seen["messages"] = list(messages)
        seen["tools"] = tools
        return _final("done")

    parent_messages = [{"role": "system", "content": "parent prompt"}]
    ctx = _ctx(tmp_path, fake_call_llm)
    subagent.spawn_subagent(ctx, "look at a.txt", "researcher", context="start in src/")

    assert seen["messages"][0]["content"] == subagent.prompt_for_role("researcher")
    assert "look at a.txt" in seen["messages"][1]["content"]
    assert "start in src/" in seen["messages"][1]["content"]
    assert parent_messages == [{"role": "system", "content": "parent prompt"}]
    assert {s["function"]["name"] for s in seen["tools"]} == set(
        subagent.ROLE_TOOLS["researcher"]
    )


def test_spawn_subagent_tags_nested_events_with_depth_and_role(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    events: list[TurnEvent] = []
    replies = iter(
        [
            {
                "status": "success",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_files", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            },
            _final("done"),
        ]
    )

    def fake_call_llm(messages, override_config=None, tools=None):
        return next(replies)

    subagent.spawn_subagent(
        _ctx(tmp_path, fake_call_llm, on_event=events.append), "read it", "researcher"
    )
    tool_events = [e for e in events if e.name == "read_files"]
    assert tool_events
    assert all(e.depth == 1 and e.agent == "researcher" for e in tool_events)


def test_spawn_subagent_refuses_to_nest(tmp_path):
    result = subagent.spawn_subagent(_ctx(tmp_path, _once(_final("x")), depth=1), "go", "coder")
    assert result["exit_code"] == 1
    assert "cannot spawn" in result["stderr"]


def test_spawn_subagent_rejects_an_empty_task(tmp_path):
    result = subagent.spawn_subagent(_ctx(tmp_path, _once(_final("x"))), "   ")
    assert result["exit_code"] == 1
    assert "task" in result["stderr"]


def test_spawn_subagent_reports_a_failing_llm_on_stderr(tmp_path):
    ctx = _ctx(tmp_path, _once({"status": "offline", "content": "[LM Studio Offline] nope"}))
    result = subagent.spawn_subagent(ctx, "go", "researcher")
    assert result["exit_code"] == 1
    assert "Offline" in result["stderr"]


def test_spawn_subagent_reports_an_exhausted_budget(tmp_path):
    (tmp_path / "a.txt").write_text("hi")

    def fake_call_llm(messages, override_config=None, tools=None):
        return {
            "status": "success",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_files", "arguments": '{"path": "a.txt"}'},
                }
            ],
        }

    result = subagent.spawn_subagent(
        _ctx(tmp_path, fake_call_llm), "loop forever", "researcher", max_steps=2
    )
    assert result["exit_code"] == 1
    assert "stopped at 2 steps" in result["stdout"]


def test_spawn_subagent_max_steps_lowers_but_cannot_raise_the_budget(tmp_path):
    seen = {}

    def fake_call_llm(messages, override_config=None, tools=None):
        seen.setdefault("calls", 0)
        seen["calls"] += 1
        return _final("done")

    result = subagent.spawn_subagent(
        _ctx(tmp_path, fake_call_llm), "go", "researcher", max_steps=99
    )
    assert "1/8 steps" in result["stdout"]


def test_spawn_subagent_applies_the_configured_model_for_that_role(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[subagent]\nmax_steps = 8\n\n"
        '[subagent.models]\nresearcher = "cheap"\n\n'
        '[models.cheap]\nname = "tiny/model"\nbase_url = "http://localhost:1234/v1"\n'
    )
    monkeypatch.setattr(agent_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(agent_config, "ENV_FILE", tmp_path / ".env")
    seen = {}

    def fake_call_llm(messages, override_config=None, tools=None):
        seen["override"] = override_config
        return _final("done")

    subagent.spawn_subagent(_ctx(tmp_path, fake_call_llm), "go", "researcher")
    assert seen["override"] == {"model": {"name": "tiny/model", "base_url": "http://localhost:1234/v1"}}


def test_dispatch_routes_spawn_subagent(tmp_path):
    from ratchet.agent import tools as agent_tools

    ctx = _ctx(tmp_path, _once(_final("delegated")))
    result = agent_tools._dispatch("spawn_subagent", {"task": "go", "role": "researcher"}, ctx)
    assert result["exit_code"] == 0
    assert "delegated" in result["stdout"]


def test_subagent_defaults_come_from_the_config_module():
    assert agent_config.DEFAULT_CONFIG["subagent"]["max_steps"] == 8
    assert agent_config.DEFAULT_CONFIG["subagent"]["max_depth"] == 1


def test_read_only_roles_are_parallel_safe():
    assert subagent.is_parallel_safe("researcher")
    assert subagent.is_parallel_safe("tester")


def test_roles_that_can_write_are_not_parallel_safe():
    assert not subagent.is_parallel_safe("coder")
    assert not subagent.is_parallel_safe("generalist")


def test_an_unknown_role_is_not_parallel_safe():
    assert not subagent.is_parallel_safe("wizard")


def test_max_parallel_default_comes_from_the_config_module():
    assert agent_config.DEFAULT_CONFIG["subagent"]["max_parallel"] == 4


def test_both_spawn_tools_are_registered():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert {subagent.SPAWN_TOOL, subagent.PARALLEL_SPAWN_TOOL} <= names


def test_the_parallel_spawn_tool_only_offers_parallel_safe_roles():
    schema = next(
        s["function"]
        for s in TOOL_SCHEMAS
        if s["function"]["name"] == subagent.PARALLEL_SPAWN_TOOL
    )
    roles = set(schema["parameters"]["properties"]["role"]["enum"])
    assert roles == {"researcher", "tester"}
    assert schema["parameters"]["required"] == ["task"]


def test_no_role_is_offered_either_spawn_tool():
    for role in subagent.ROLE_TOOLS:
        names = {s["function"]["name"] for s in subagent.schema_for_role(role)}
        assert not names & {subagent.SPAWN_TOOL, subagent.PARALLEL_SPAWN_TOOL}


def test_a_parallel_spawn_refuses_a_role_that_can_write(tmp_path):
    result = subagent.spawn_subagent(
        _ctx(tmp_path, _once(_final("x"))), "go", "coder", parallel=True
    )
    assert result["exit_code"] == 1
    assert "coder" in result["stderr"]


def test_a_parallel_spawn_accepts_a_read_only_role(tmp_path):
    result = subagent.spawn_subagent(
        _ctx(tmp_path, _once(_final("found it"))), "go", "researcher", parallel=True
    )
    assert result["exit_code"] == 0


def test_dispatch_routes_the_parallel_spawn_tool(tmp_path):
    from ratchet.agent import tools as agent_tools

    ctx = _ctx(tmp_path, _once(_final("delegated")))
    result = agent_tools._dispatch(
        subagent.PARALLEL_SPAWN_TOOL, {"task": "go", "role": "researcher"}, ctx
    )
    assert result["exit_code"] == 0
    assert "delegated" in result["stdout"]


def test_dispatch_refuses_a_writing_role_on_the_parallel_spawn_tool(tmp_path):
    from ratchet.agent import tools as agent_tools

    ctx = _ctx(tmp_path, _once(_final("delegated")))
    result = agent_tools._dispatch(
        subagent.PARALLEL_SPAWN_TOOL, {"task": "go", "role": "coder"}, ctx
    )
    assert result["exit_code"] == 1
