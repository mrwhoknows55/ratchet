from ratchet.agent import plan
from ratchet.agent import tools as agent_tools


def test_create_plan_writes_a_checklist_to_plan_md(tmp_path):
    output, exit_code = agent_tools.execute_tool_result(
        "create_plan",
        {"title": "Math utils", "steps": ["Add safe_div", "Add tests"]},
        tmp_path,
    )
    assert exit_code == 0
    assert plan.PLAN_FILE in output
    assert (tmp_path / plan.PLAN_FILE).read_text() == (
        "# Math utils\n\n- [ ] Add safe_div\n- [ ] Add tests\n"
    )


def test_create_plan_rejects_a_plan_without_steps(tmp_path):
    output, exit_code = agent_tools.execute_tool_result(
        "create_plan", {"title": "Empty", "steps": ["  "]}, tmp_path
    )
    assert exit_code == 1
    assert "step" in output
    assert not (tmp_path / plan.PLAN_FILE).exists()


def test_plan_mode_schemas_exclude_every_tool_that_changes_state():
    names = {schema["function"]["name"] for schema in plan.plan_mode_schemas()}
    assert "create_plan" in names
    assert "read_files" in names
    assert not names & {"write_files", "delete_file", "run_command", "spawn_subagent"}
