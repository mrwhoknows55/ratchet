import json
from pathlib import Path
from typing import Callable

from ratchet.shell.executor import delete_file, list_files, read_files, search_files, write_files

MAX_TOOL_ITERATIONS = 5

_PATH_PROPERTY = {"path": {"type": "string", "description": "Path relative to the sandbox root."}}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the sandboxed working directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents in the sandbox for a pattern (uses rg/grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Pattern to search for."}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read the contents of a file in the sandboxed working directory.",
            "parameters": {"type": "object", "properties": _PATH_PROPERTY, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_files",
            "description": "Write content to a file in the sandbox, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_PATH_PROPERTY,
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file in the sandboxed working directory.",
            "parameters": {"type": "object", "properties": _PATH_PROPERTY, "required": ["path"]},
        },
    },
]


def execute_tool(name: str, arguments: dict, sandbox_root: Path) -> str:
    if name == "list_files":
        result = list_files(sandbox_root)
    elif name == "search_files":
        result = search_files(sandbox_root, arguments["pattern"])
    elif name == "read_files":
        result = read_files(sandbox_root, arguments["path"])
    elif name == "write_files":
        result = write_files(sandbox_root, arguments["path"], arguments["content"])
    elif name == "delete_file":
        result = delete_file(sandbox_root, arguments["path"])
    else:
        return f"Error: unknown tool '{name}'"
    return (str(result["stdout"]) + str(result["stderr"])).strip() or "(no output)"


def run_agent_turn(
    call_llm_fn: Callable,
    user_text: str,
    sandbox_root: Path,
    override_config: dict | None = None,
    on_tool_call: Callable[[str], None] | None = None,
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
            tool_name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"] or "{}")
            if on_tool_call:
                on_tool_call(f"tool: {tool_name} running...")
            output = execute_tool(tool_name, arguments, sandbox_root)
            if on_tool_call:
                on_tool_call(f"tool: {tool_name} -> {output}")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )

    return "[Error] tool call loop exceeded max iterations"
