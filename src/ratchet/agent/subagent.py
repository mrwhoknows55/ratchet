from pathlib import Path

from ratchet.agent.tools import TOOL_SCHEMAS

ROLE_PROMPT_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "roles"

DEFAULT_ROLE = "generalist"

_READ_TOOLS = [
    "list_files",
    "search_files",
    "file_search",
    "read_files",
    "read_file_range",
    "get_file_info",
]

ROLE_TOOLS = {
    "researcher": [*_READ_TOOLS, "check_command", "search_web", "fetch_url"],
    "coder": [
        *_READ_TOOLS,
        "write_files",
        "replace_in_file",
        "append_file",
        "delete_file",
        "copy_file",
        "move_file",
        "rollback_file",
    ],
    "tester": [*_READ_TOOLS, "run_command", "check_command"],
    "generalist": [schema["function"]["name"] for schema in TOOL_SCHEMAS],
}


def schema_for_role(role: str) -> list[dict]:
    allowed = set(ROLE_TOOLS.get(role, ROLE_TOOLS[DEFAULT_ROLE]))
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]


def prompt_for_role(role: str) -> str:
    name = role if role in ROLE_TOOLS else DEFAULT_ROLE
    return (ROLE_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
