import json
from pathlib import Path
from typing import Callable

from ratchet.shell.executor import list_files

MAX_TOOL_ITERATIONS = 5

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the sandboxed working directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def execute_tool(name: str, arguments: dict, sandbox_root: Path) -> str:
    if name == "list_files":
        result = list_files(sandbox_root)
        return (str(result["stdout"]) + str(result["stderr"])).strip() or "(no output)"
    return f"Error: unknown tool '{name}'"


def run_agent_turn(
    call_llm_fn: Callable,
    user_text: str,
    sandbox_root: Path,
    override_config: dict | None = None,
) -> str:
    messages: list[dict] = [{"role": "user", "content": user_text}]

    for _ in range(MAX_TOOL_ITERATIONS):
        result = call_llm_fn(messages, override_config, tools=TOOL_SCHEMAS)
        if result["status"] != "success":
            return result["content"]

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            return result["content"]

        messages.append(
            {"role": "assistant", "content": result.get("content") or "", "tool_calls": tool_calls}
        )
        for call in tool_calls:
            arguments = json.loads(call["function"]["arguments"] or "{}")
            output = execute_tool(call["function"]["name"], arguments, sandbox_root)
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )

    return "[Error] tool call loop exceeded max iterations"
