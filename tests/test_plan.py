import json

import pytest

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
    assert not names & {"write_files", "delete_file", "run_command"}
    assert "get_plan" in names
    assert "update_plan_step" not in names


def test_planning_spawn_schemas_only_offer_allowed_roles_without_changing_act_schemas():
    schemas = {s["function"]["name"]: s["function"] for s in plan.plan_mode_schemas()}
    roles = schemas["spawn_subagent"]["parameters"]["properties"]["role"]["enum"]
    assert set(roles) == {"planner", "researcher"}
    roles = schemas["spawn_parallel_subagent"]["parameters"]["properties"]["role"]["enum"]
    assert roles == ["researcher"]
    original = next(s["function"] for s in agent_tools.TOOL_SCHEMAS
                    if s["function"]["name"] == "spawn_subagent")
    assert "coder" in original["parameters"]["properties"]["role"]["enum"]


def test_structured_plan_round_trips_through_tools(tmp_path):
    output, code = agent_tools.execute_tool_result(
        "create_plan",
        {
            "title": " Math utils ",
            "summary": " Keep division safe. ",
            "risks": ["Zero denominators"],
            "steps": [
                {"id": "divide", "description": " Add safe_div ",
                 "target_files": ["math_ops.py"], "notes": "Reject zero"},
                "Add tests",
            ],
        },
        tmp_path,
    )
    assert code == 0, output
    stored = json.loads((tmp_path / "PLAN.json").read_text())
    assert stored == {
        "title": "Math utils",
        "summary": "Keep division safe.",
        "risks": ["Zero denominators"],
        "steps": [
            {"id": "divide", "description": "Add safe_div", "target_files": ["math_ops.py"],
             "notes": "Reject zero", "status": "pending"},
            {"id": 2, "description": "Add tests", "target_files": [],
             "notes": "", "status": "pending"},
        ],
    }
    output, code = agent_tools.execute_tool_result("get_plan", {}, tmp_path)
    assert code == 0
    payload = json.loads(output)
    assert payload["plan"] == stored
    assert payload["progress"] == {"total": 2, "completed": 0, "percent": 0.0}
    markdown = (tmp_path / "PLAN.md").read_text()
    assert "Keep division safe." in markdown
    assert "math_ops.py" in markdown
    assert "Reject zero" in markdown
    assert "Zero denominators" in markdown


@pytest.mark.parametrize("status", ["in_progress", "completed", "failed", "skipped", "pending"])
def test_update_step_persists_status_notes_and_progress(tmp_path, status):
    plan.create_plan(tmp_path, "Math", ["Implement", "Test"])
    output, code = agent_tools.execute_tool_result(
        "update_plan_step", {"step_id": "1", "status": status, "notes": "Verified"}, tmp_path
    )
    assert code == 0, output
    payload = json.loads(output)
    step = payload["plan"]["steps"][0]
    assert step["status"] == status
    assert step["notes"] == "Verified"
    assert payload["plan"]["steps"][1]["status"] == "pending"
    assert payload["progress"]["percent"] == (50.0 if status == "completed" else 0.0)
    assert json.loads((tmp_path / "PLAN.json").read_text()) == payload["plan"]
    markdown = (tmp_path / "PLAN.md").read_text()
    assert "Verified" in markdown
    assert ("- [x] Implement" in markdown) == (status == "completed")
    if status != "pending":
        assert status in markdown


def test_update_step_preserves_notes_when_omitted(tmp_path):
    plan.create_plan(tmp_path, "Math", [{"description": "Implement", "notes": "Keep this"}])
    output, code = agent_tools.execute_tool_result(
        "update_plan_step", {"step_id": 1, "status": "completed"}, tmp_path
    )
    assert code == 0
    assert json.loads(output)["plan"]["steps"][0]["notes"] == "Keep this"


@pytest.mark.parametrize("arguments", [
    {"step_id": 9, "status": "completed"},
    {"step_id": 1, "status": "done"},
    {"step_id": 1, "status": "completed", "notes": []},
])
def test_invalid_updates_leave_both_plan_files_unchanged(tmp_path, arguments):
    plan.create_plan(tmp_path, "Math", ["Implement"])
    before = {name: (tmp_path / name).read_bytes() for name in ("PLAN.md", "PLAN.json")}
    output, code = agent_tools.execute_tool_result("update_plan_step", arguments, tmp_path)
    assert code == 1
    assert "Error" in output
    assert before == {name: (tmp_path / name).read_bytes() for name in before}


@pytest.mark.parametrize("steps", [
    [{"description": ""}],
    [{"id": 1, "description": "One"}, {"id": "1", "description": "Two"}],
    [42],
    "Implement",
    [{"description": "One", "target_files": "math.py"}],
    [{"description": "One", "status": "done"}],
])
def test_invalid_structured_steps_do_not_replace_existing_plan(tmp_path, steps):
    plan.create_plan(tmp_path, "Math", ["Implement"])
    before = (tmp_path / "PLAN.md").read_bytes()
    output, code = agent_tools.execute_tool_result(
        "create_plan", {"title": "Replacement", "steps": steps}, tmp_path
    )
    assert code == 1
    assert "Error" in output
    assert (tmp_path / "PLAN.md").read_bytes() == before


@pytest.mark.parametrize("content", [None, "{broken", '{"steps": []}'])
def test_get_and_update_report_missing_or_corrupt_plan(tmp_path, content):
    if content is not None:
        (tmp_path / "PLAN.json").write_text(content)
    for name, arguments in [
        ("get_plan", {}),
        ("update_plan_step", {"step_id": 1, "status": "completed"}),
    ]:
        output, code = agent_tools.execute_tool_result(name, arguments, tmp_path)
        assert code == 1
        assert "Error" in output
    assert not (tmp_path / "PLAN.md").exists()


@pytest.mark.parametrize("filename", ["PLAN.md", "PLAN.json"])
def test_create_plan_rejects_symlinks_outside_sandbox(tmp_path, filename):
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Keep this")
    (root / filename).symlink_to(outside)
    output, code = agent_tools.execute_tool_result(
        "create_plan", {"title": "Math", "steps": ["Implement"]}, root
    )
    assert code == 1
    assert "Error" in output
    assert outside.read_text() == "Keep this"
    assert not (root / ({"PLAN.md", "PLAN.json"} - {filename}).pop()).exists()
