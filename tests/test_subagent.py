import pytest

from ratchet.agent import subagent
from ratchet.agent.tools import TOOL_SCHEMAS

ALL_TOOLS = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
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
