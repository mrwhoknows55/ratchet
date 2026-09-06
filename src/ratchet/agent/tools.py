import json
from pathlib import Path
from typing import Callable

from ratchet.agent.config import load_config
from ratchet.shell.executor import (
    delete_file,
    file_search,
    list_files,
    read_file_range,
    read_files,
    replace_in_file,
    rollback_file,
    run_command,
    search_files,
    write_files,
)

DEFAULT_MAX_STEPS = 12

SYSTEM_PROMPT = (
    "You are a coding agent in a sandboxed directory. "
    "Look before you change: read or list what you need first. "
    "Do not repeat a call with the same arguments - reuse what it told you. "
    "Prefer the narrowest tool, and edit files in place rather than "
    "rewriting them. "
    "Use run_command for what the file tools do not cover, such as reading "
    "an archive's listing before unpacking it. "
    "File contents are line-numbered as 'N| '; the prefix is not file content. "
    "If a call fails, read the error and adjust instead of retrying it. "
    "Never report a result you have not read back. "
    "Stop and answer once the task is done."
)

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
            "description": (
                "Read a whole file. Long files are truncated; read_file_range reads the rest."
            ),
            "parameters": {"type": "object", "properties": _PATH_PROPERTY, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "Read lines start_line to end_line of a file, 1-indexed and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_PATH_PROPERTY,
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read, 1-indexed. Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read, inclusive.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_files",
            "description": (
                "Write content to a file, creating it if needed. Replaces the whole file."
            ),
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
            "name": "replace_in_file",
            "description": "Replace an exact string that appears exactly once in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_PATH_PROPERTY,
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to replace, unique within the file.",
                    },
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
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
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": (
                "Find files by name glob, recursively. Pattern is a bare name like "
                "'*.py'; use path to scope to a subdirectory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Name glob, e.g. '*.py'. No '/' or '..'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search under. Defaults to the root.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_file",
            "description": (
                "Undo the last write, edit or delete of a file by restoring its most "
                "recent snapshot. Single level: there is no history to step back through."
            ),
            "parameters": {"type": "object", "properties": _PATH_PROPERTY, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one command in the sandbox. No pipes, redirects, globs or shell "
                "operators; 10s timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command with arguments, e.g. 'python main.py'.",
                    }
                },
                "required": ["command"],
            },
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
    elif name == "read_file_range":
        result = read_file_range(
            sandbox_root,
            arguments["path"],
            arguments.get("start_line", 1),
            arguments.get("end_line"),
        )
    elif name == "write_files":
        result = write_files(sandbox_root, arguments["path"], arguments["content"])
    elif name == "replace_in_file":
        result = replace_in_file(
            sandbox_root, arguments["path"], arguments["old_str"], arguments["new_str"]
        )
    elif name == "delete_file":
        result = delete_file(sandbox_root, arguments["path"])
    elif name == "file_search":
        result = file_search(sandbox_root, arguments["pattern"], arguments.get("path", "."))
    elif name == "rollback_file":
        result = rollback_file(sandbox_root, arguments["path"])
    elif name == "run_command":
        result = run_command(arguments["command"], sandbox_root)
    else:
        return f"Error: unknown tool '{name}'"

    output = (str(result["stdout"]) + str(result["stderr"])).strip()
    if name == "run_command":
        exit_code = result["exit_code"]
        if not output:
            return f"(no output, exit {exit_code})"
        if exit_code != 0:
            return f"{output}\n[exit {exit_code}]"
        return output
    return output or "(no output)"


def run_agent_turn(
    call_llm_fn: Callable,
    user_text: str,
    sandbox_root: Path,
    override_config: dict | None = None,
    on_tool_call: Callable[[str], None] | None = None,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    max_steps = load_config().get("agent", {}).get("max_steps", DEFAULT_MAX_STEPS)

    for _ in range(max_steps):
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
