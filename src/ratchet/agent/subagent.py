from pathlib import Path

from ratchet.agent.config import load_config
from ratchet.agent.context import AgentContext
from ratchet.agent.events import TurnResult
from ratchet.agent.tools import TOOL_SCHEMAS

ROLE_PROMPT_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "roles"

DEFAULT_ROLE = "generalist"
SPAWN_TOOL = "spawn_subagent"
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_DEPTH = 1

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
    "generalist": [
        schema["function"]["name"]
        for schema in TOOL_SCHEMAS
        if schema["function"]["name"] != SPAWN_TOOL
    ],
}


def normalize_role(role: str) -> str:
    name = (role or "").strip().lower()
    return name if name in ROLE_TOOLS else DEFAULT_ROLE


def schema_for_role(role: str) -> list[dict]:
    allowed = set(ROLE_TOOLS.get(role, ROLE_TOOLS[DEFAULT_ROLE]))
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]


def prompt_for_role(role: str) -> str:
    name = role if role in ROLE_TOOLS else DEFAULT_ROLE
    return (ROLE_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def _subagent_config() -> dict:
    return load_config().get("subagent", {})


def _model_override(role: str) -> dict | None:
    alias = _subagent_config().get("models", {}).get(role, "")
    if not alias:
        return None
    model = load_config().get("models", {}).get(alias)
    return {"model": model} if model else None


def _error(message: str) -> dict[str, str | int]:
    return {"stdout": "", "stderr": message, "exit_code": 1}


def _metadata_line(role: str, result: TurnResult, budget: int) -> str:
    parts = [role, f"{result.elapsed:.1f}s", f"{result.steps}/{budget} steps"]
    if result.prompt_tokens is not None:
        parts.append(f"{result.prompt_tokens + (result.completion_tokens or 0)} tok")
    if result.tools_used:
        parts.append("via " + ", ".join(dict.fromkeys(result.tools_used)))
    return "[" + " \u00b7 ".join(parts) + "]"


def spawn_subagent(
    ctx: AgentContext,
    task: str,
    role: str = DEFAULT_ROLE,
    context: str = "",
    max_steps: int | None = None,
) -> dict[str, str | int]:
    clean_task = (task or "").strip()
    if not clean_task:
        return _error("Error: 'task' must not be empty.")

    max_depth = int(_subagent_config().get("max_depth", DEFAULT_MAX_DEPTH))
    if ctx.depth >= max_depth:
        return _error(
            f"Error: subagents cannot spawn subagents (depth {ctx.depth} of {max_depth})."
        )
    if ctx.call_llm_fn is None:
        return _error("Error: no LLM client is available to run a subagent.")

    role_name = normalize_role(role)

    budget = int(_subagent_config().get("max_steps", DEFAULT_MAX_STEPS))
    if max_steps:
        budget = max(1, min(int(max_steps), budget))

    user_text = clean_task
    if context and context.strip():
        user_text += f"\n\nContext from the parent agent:\n{context.strip()}"

    from ratchet.agent.loop import run_turn

    result = run_turn(
        ctx.call_llm_fn,
        user_text,
        ctx.sandbox_root,
        _model_override(role_name) or ctx.override_config,
        ctx.on_event,
        [{"role": "system", "content": prompt_for_role(role_name)}],
        tools_schema=schema_for_role(role_name),
        max_steps=budget,
        depth=ctx.depth + 1,
        agent=role_name,
    )

    meta = _metadata_line(role_name, result, budget)
    summary = (result.text or "").strip()

    if result.status == "error":
        return {"stdout": "", "stderr": f"{summary}\n{meta}".strip(), "exit_code": 1}
    if result.status == "max_steps":
        note = f"Subagent stopped at {result.steps} steps without finishing."
        return {"stdout": f"{note}\n{meta}", "stderr": "", "exit_code": 1}
    return {"stdout": f"{summary}\n{meta}", "stderr": "", "exit_code": 0}
