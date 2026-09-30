from pathlib import Path

PLAN_FILE = "PLAN.md"

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
}

PLAN_MODE_NOTE = (
    "[Plan mode] Read-only: you cannot edit files, run commands or spawn subagents. "
    "Inspect what you need, then save the plan with create_plan and summarise it."
)

EXECUTE_PLAN_PROMPT = (
    f"Execute the plan in {PLAN_FILE}. Work through the unchecked steps in order. "
    "After finishing each step, tick it by replacing its '- [ ]' with '- [x]' "
    "using replace_in_file. Stop and report if a step fails."
)


def plan_mode_schemas() -> list[dict]:
    from ratchet.agent.tools import TOOL_SCHEMAS

    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in PLAN_MODE_TOOLS]


def blocked_in_plan_mode(name: str) -> str:
    return (
        f"[Error] {name} is blocked in plan mode; "
        "only read-only tools and create_plan are allowed"
    )


def create_plan(root: Path, title: str, steps: list[str]) -> dict[str, str | int]:
    clean_steps = [step.strip() for step in steps if step and step.strip()]
    if not clean_steps:
        return {"stdout": "", "stderr": "Error: a plan needs at least one step.", "exit_code": 1}
    lines = [f"# {title.strip() or 'Plan'}", "", *(f"- [ ] {step}" for step in clean_steps)]
    (root / PLAN_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "stdout": f"Saved {len(clean_steps)} steps to '{PLAN_FILE}'",
        "stderr": "",
        "exit_code": 0,
    }
