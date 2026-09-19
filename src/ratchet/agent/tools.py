import json
import platform
from pathlib import Path

from ratchet.agent.config import load_config
from ratchet.agent.context import AgentContext, as_context
from ratchet.agent.web import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_LIMIT,
    fetch_url,
    search_web,
)
from ratchet.shell.executor import (
    DEFAULT_COMMAND_TIMEOUT,
    MAX_COMMAND_TIMEOUT,
    append_file,
    check_command,
    copy_file,
    delete_file,
    file_search,
    get_file_info,
    list_files,
    move_file,
    read_file_range,
    read_files,
    replace_in_file,
    rollback_file,
    run_command,
    search_files,
    write_files,
)


def _command_timeout() -> int:
    return load_config().get("agent", {}).get("command_timeout", DEFAULT_COMMAND_TIMEOUT)

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "system.md"
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

_PLATFORM_NAMES = {"Darwin": "macOS"}


def detect_platform() -> str:
    system = platform.system()
    return _PLATFORM_NAMES.get(system, system)


def render_system_prompt(platform_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.replace("{platform}", platform_name)


SYSTEM_PROMPT = render_system_prompt(detect_platform())

_PATH_PROPERTY = {"path": {"type": "string", "description": "Path relative to the sandbox root."}}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List a directory in the sandbox, with a size for each file and a "
                "trailing '/' on each subdirectory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list. Defaults to the sandbox root.",
                    }
                },
                "required": [],
            },
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
                    "pattern": {"type": "string", "description": "Pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": "Directory to search under. Defaults to the sandbox root.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": (
                "Report a file's size, line count, modified time and sha256 without "
                "reading its contents. Line count is -1 for binary files."
            ),
            "parameters": {"type": "object", "properties": _PATH_PROPERTY, "required": ["path"]},
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
            "name": "append_file",
            "description": (
                "Append text to the end of a file, creating it if needed. Use this "
                "instead of rewriting the whole file to add to it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_PATH_PROPERTY,
                    "content": {"type": "string", "description": "Text to append."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Delete a file, or a directory when recursive is true. Contents are "
                "snapshotted first, so rollback_file can bring a deleted file back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_PATH_PROPERTY,
                    "recursive": {
                        "type": "boolean",
                        "description": "Required to delete a directory and everything inside it.",
                    },
                },
                "required": ["path"],
            },
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
            "name": "copy_file",
            "description": (
                "Copy a file, or a whole directory tree, to another path in the sandbox. "
                "Missing parent directories are created."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Path to copy from."},
                    "destination": {"type": "string", "description": "Path to copy to."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": (
                "Move or rename a file or directory within the sandbox. Missing parent "
                "directories are created."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Path to move from."},
                    "destination": {"type": "string", "description": "Path to move to."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the live web and return ranked titles, URLs and snippets. "
                "Use fetch_url to read a result in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"How many results to return. Defaults to {DEFAULT_MAX_RESULTS}, "
                            f"capped at {MAX_RESULTS_LIMIT}."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch one web page and return its content as Markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http:// or https:// URL.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one command in the sandbox. No pipes, redirects or globs - "
                "write a script and run it. Raise timeout for slow work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command with arguments, e.g. 'python main.py'.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to wait before killing the command. Defaults to "
                            f"{DEFAULT_COMMAND_TIMEOUT}, capped at {MAX_COMMAND_TIMEOUT}."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_command",
            "description": (
                "Check whether a command or binary is installed and on PATH, "
                "before assuming it isn't. Returns its resolved path if found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Command name, e.g. 'yt-dlp'."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": (
                "Delegate a self-contained sub-task to a focused subagent and get back "
                "only its summary, keeping long lookups out of this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "What the subagent must do, stated in full - it cannot see "
                            "this conversation."
                        ),
                    },
                    "role": {
                        "type": "string",
                        "enum": ["researcher", "coder", "tester", "generalist"],
                        "description": (
                            "researcher reads, searches and browses; coder edits files; "
                            "tester runs commands; generalist has every tool."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": "Background it needs, such as paths you already found.",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Lower the subagent's step budget. Cannot raise it.",
                    },
                },
                "required": ["task"],
            },
        },
    },
]


def _dispatch(name: str, arguments: dict, ctx: AgentContext) -> dict[str, str | int]:
    sandbox_root = ctx.sandbox_root
    if name == "list_files":
        result = list_files(sandbox_root, arguments.get("path", "."))
    elif name == "search_files":
        result = search_files(
            sandbox_root, arguments["pattern"], arguments.get("path", ".")
        )
    elif name == "get_file_info":
        result = get_file_info(sandbox_root, arguments["path"])
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
    elif name == "append_file":
        result = append_file(sandbox_root, arguments["path"], arguments["content"])
    elif name == "delete_file":
        result = delete_file(
            sandbox_root, arguments["path"], arguments.get("recursive", False)
        )
    elif name == "copy_file":
        result = copy_file(sandbox_root, arguments["source"], arguments["destination"])
    elif name == "move_file":
        result = move_file(sandbox_root, arguments["source"], arguments["destination"])
    elif name == "search_web":
        result = search_web(
            arguments["query"], arguments.get("max_results", DEFAULT_MAX_RESULTS)
        )
    elif name == "fetch_url":
        result = fetch_url(arguments["url"])
    elif name == "file_search":
        result = file_search(sandbox_root, arguments["pattern"], arguments.get("path", "."))
    elif name == "rollback_file":
        result = rollback_file(sandbox_root, arguments["path"])
    elif name == "run_command":
        result = run_command(
            arguments["command"], sandbox_root, arguments.get("timeout", _command_timeout())
        )
    elif name == "check_command":
        result = check_command(arguments["name"])
    elif name == "spawn_subagent":
        from ratchet.agent.subagent import spawn_subagent

        result = spawn_subagent(
            ctx,
            arguments.get("task", ""),
            arguments.get("role", ""),
            arguments.get("context", ""),
            arguments.get("max_steps"),
        )
    else:
        return {
            "stdout": "",
            "stderr": f"Error: unknown tool '{name}'",
            "exit_code": 1,
        }
    return result


def execute_tool_result(
    name: str, arguments: dict, context: AgentContext | Path
) -> tuple[str, int]:
    result = _dispatch(name, arguments, as_context(context))
    exit_code = int(result["exit_code"])
    output = (str(result["stdout"]) + str(result["stderr"])).strip()
    if name == "run_command":
        if not output:
            return f"(no output, exit {exit_code})", exit_code
        if exit_code != 0:
            return f"{output}\n[exit {exit_code}]", exit_code
        return output, exit_code
    return output or "(no output)", exit_code


def execute_tool(name: str, arguments: dict, context: AgentContext | Path) -> str:
    return execute_tool_result(name, arguments, context)[0]


def save_session(messages: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages), encoding="utf-8")


def load_session(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def clear_session(path: Path) -> None:
    path.unlink(missing_ok=True)
