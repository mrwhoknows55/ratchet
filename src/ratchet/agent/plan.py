import json
from copy import deepcopy
from pathlib import Path

PLAN_FILE = "PLAN.md"
PLAN_JSON_FILE = "PLAN.json"
PLAN_STATUSES = ("pending", "in_progress", "completed", "skipped", "failed")
PLAN_DELEGATION_ROLES = ("researcher", "planner")

PLAN_MODE_TOOLS = {
    "list_files",
    "search_files",
    "file_search",
    "read_files",
    "read_file_range",
    "get_file_info",
    "check_command",
    "search_web",
    "fetch_url",
    "create_plan",
    "get_plan",
    "spawn_subagent",
    "spawn_parallel_subagent",
    "ask_user_question",
}

PLAN_MODE_NOTE = (
    "[Plan mode] Read-only: you cannot edit workspace files or run commands. "
    "Delegate inspection to researcher or planner subagents; only researchers may run "
    "in parallel. Planners may save plans but cannot execute them. "
    "Use ask_user_question to resolve requirements that materially change the design "
    "before saving the plan. If no answer is available, report the unresolved question. "
    "Inspect what you need, then save the plan with create_plan and summarise it."
)

EXECUTE_PLAN_PROMPT = (
    "Retrieve the active plan with get_plan. Stop and report if no valid plan exists. "
    "Work through pending steps in order, using update_plan_step to mark each in_progress "
    "before starting and completed only after verification. Inspect any in_progress step "
    "before resuming it. Stop and report failed steps, recording failure notes with "
    "update_plan_step. Do not edit the plan files directly."
)

PLAN_EXECUTION_NOTE = (
    "[Build mode] Workspace edits and commands are enabled; "
    "previous planning restrictions no longer apply. "
    "An active structured plan exists. For requested work covered by that plan, "
    f"{EXECUTE_PLAN_PROMPT} "
    "Track delegated work too: verify subagent results before completing its plan step. "
    "Before reporting completion, retrieve the plan again and reconcile step statuses "
    "with verified results. Do not execute unrelated plan steps."
)


def plan_mode_schemas() -> list[dict]:
    from ratchet.agent.tools import TOOL_SCHEMAS

    schemas = deepcopy([s for s in TOOL_SCHEMAS if s["function"]["name"] in PLAN_MODE_TOOLS])
    for schema in schemas:
        function = schema["function"]
        if function["name"] == "spawn_subagent":
            function["parameters"]["properties"]["role"]["enum"] = list(PLAN_DELEGATION_ROLES)
            function["parameters"]["required"] = ["task", "role"]
        elif function["name"] == "spawn_parallel_subagent":
            function["parameters"]["properties"]["role"]["enum"] = ["researcher"]
            function["parameters"]["required"] = ["task", "role"]
    return schemas


def blocked_in_plan_mode(name: str) -> str:
    return (
        f"[Error] {name} is blocked in plan mode; "
        "only read-only tools and create_plan are allowed"
    )


def _error(error: Exception | str) -> dict[str, str | int]:
    return {"stdout": "", "stderr": f"Error: {error}", "exit_code": 1}


def _text(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    return value.strip()


def _strings(value: list[str], field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings.")
    return [_text(item, field) for item in value]


def _normalize_plan(title: str, steps: list, summary: str, risks: list[str]) -> dict:
    if not isinstance(steps, list) or not steps:
        raise ValueError("a plan needs at least one step.")
    normalized = []
    ids = set()
    for index, raw in enumerate(steps, 1):
        if isinstance(raw, str):
            raw = {"description": raw}
        if not isinstance(raw, dict):
            raise ValueError(f"Step {index} must be a string or object.")
        description = _text(raw.get("description", ""), "description")
        if not description:
            raise ValueError(f"Plan step {index} needs a description.")
        step_id = raw.get("id", index)
        if type(step_id) not in (int, str) or not str(step_id).strip():
            raise ValueError(f"Step {index} needs an integer or non-empty string ID.")
        if isinstance(step_id, str):
            step_id = step_id.strip()
        if str(step_id) in ids:
            raise ValueError(f"Duplicate step ID: {step_id}.")
        ids.add(str(step_id))
        status = raw.get("status", "pending")
        if status not in PLAN_STATUSES:
            raise ValueError(f"Invalid step status: {status}.")
        normalized.append({
            "id": step_id,
            "description": description,
            "status": status,
            "target_files": _strings(raw.get("target_files", []), "target_files"),
            "notes": _text(raw.get("notes", ""), "notes"),
        })
    return {
        "title": _text(title, "title") or "Plan",
        "summary": _text(summary, "summary"),
        "steps": normalized,
        "risks": _strings(risks, "risks"),
    }


def _plan_path(root: Path, filename: str) -> Path:
    path = root / filename
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Refusing unsafe plan path: {filename}.")
    if path.is_dir():
        raise ValueError(f"{filename} is a directory, not a file.")
    return path


def _markdown(data: dict) -> str:
    lines = [f"# {data['title']}", ""]
    if data["summary"]:
        lines.extend([data["summary"], ""])
    for step in data["steps"]:
        status = step["status"]
        checkbox = "x" if status == "completed" else " "
        tag = f" ({status})" if status != "pending" else ""
        lines.append(f"- [{checkbox}] {step['description']}{tag}")
        if step["target_files"]:
            lines.append("  - Files: " + ", ".join(f"`{p}`" for p in step["target_files"]))
        if step["notes"]:
            lines.append(f"  - Notes: {step['notes']}")
    if data["risks"]:
        lines.extend(["", "## Risks", *(f"- {risk}" for risk in data["risks"])])
    return "\n".join(lines) + "\n"


def _save(root: Path, data: dict) -> None:
    json_path = _plan_path(root, PLAN_JSON_FILE)
    markdown_path = _plan_path(root, PLAN_FILE)
    root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(data), encoding="utf-8")


def _load(root: Path) -> dict:
    path = _plan_path(root, PLAN_JSON_FILE)
    if not path.exists():
        raise ValueError("No active structured plan found; create a plan first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"title", "steps", "summary", "risks"} <= data.keys():
        raise ValueError("Invalid structured plan data.")
    return _normalize_plan(data["title"], data["steps"], data["summary"], data["risks"])


def _result(data: dict) -> dict[str, str | int]:
    total = len(data["steps"])
    completed = sum(step["status"] == "completed" for step in data["steps"])
    payload = {
        "plan": data,
        "progress": {"total": total, "completed": completed,
                     "percent": round(completed / total * 100, 1)},
    }
    return {"stdout": json.dumps(payload, indent=2), "stderr": "", "exit_code": 0}


def create_plan(
    root: Path, title: str, steps: list[str | dict], summary: str = "",
    risks: list[str] | None = None,
) -> dict[str, str | int]:
    try:
        data = _normalize_plan(title, steps, summary, risks if risks is not None else [])
        _save(root, data)
    except (ValueError, OSError) as error:
        return _error(error)
    return {
        "stdout": f"Saved {len(data['steps'])} steps to '{PLAN_FILE}' and '{PLAN_JSON_FILE}'",
        "stderr": "",
        "exit_code": 0,
    }


def get_plan(root: Path) -> dict[str, str | int]:
    try:
        return _result(_load(root))
    except (ValueError, OSError) as error:
        return _error(error)


def update_plan_step(
    root: Path, step_id: int | str, status: str, notes: str | None = None,
) -> dict[str, str | int]:
    try:
        if status not in PLAN_STATUSES:
            raise ValueError(f"Invalid step status: {status}.")
        if type(step_id) not in (int, str):
            raise ValueError("step_id must be an integer or string.")
        data = _load(root)
        step = next((s for s in data["steps"] if str(s["id"]) == str(step_id)), None)
        if step is None:
            raise ValueError(f"Unknown step ID: {step_id}.")
        if notes is not None:
            step["notes"] = _text(notes, "notes")
        step["status"] = status
        _save(root, data)
        return _result(data)
    except (ValueError, OSError) as error:
        return _error(error)
