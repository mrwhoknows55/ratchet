import shlex
import shutil
import subprocess
from pathlib import Path

DEFAULT_MAX_READ_LINES = 200


def _resolve_path(root: Path, relative: str) -> Path | None:
    if relative.startswith("/") or ".." in Path(relative).parts:
        return None
    return root / relative


def _number_lines(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(lines, start))


def _read_lines(root: Path, path: str) -> tuple[list[str] | None, dict[str, str | int] | None]:
    target = _resolve_path(root, path)
    if target is None:
        return None, _access_denied(path)
    if not target.is_file():
        return None, {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    try:
        return target.read_text().splitlines(), None
    except UnicodeDecodeError:
        return None, {
            "stdout": "",
            "stderr": f"Cannot read '{path}': not UTF-8 text (binary file).",
            "exit_code": 1,
        }


def _access_denied(path: str) -> dict[str, str | int]:
    return {
        "stdout": "",
        "stderr": (
            f"Access Denied: Path traversal or absolute path '{path}' forbidden outside sandbox."
        ),
        "exit_code": 1,
    }


def run_command(command: str, root: Path) -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)

    command_str = command.strip()
    if not command_str:
        return {"stdout": "", "stderr": "Error: Empty command provided.", "exit_code": 1}

    for token in command_str.split():
        if token.startswith("/") or ".." in token:
            return {
                "stdout": "",
                "stderr": (
                    f"Access Denied: Path traversal or absolute path '{token}' "
                    "forbidden outside sandbox."
                ),
                "exit_code": 1,
            }

    args = shlex.split(command_str)
    try:
        result = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: '{args[0]}'", "exit_code": 127}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "exit_code": 1}


def list_files(root: Path) -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(entry.name for entry in root.iterdir())
    return {"stdout": "\n".join(names), "stderr": "", "exit_code": 0}


def read_files(root: Path, path: str) -> dict[str, str | int]:
    lines, error = _read_lines(root, path)
    if error is not None:
        return error
    total = len(lines)
    if total > DEFAULT_MAX_READ_LINES:
        body = _number_lines(lines[:DEFAULT_MAX_READ_LINES], 1)
        footer = (
            f"[truncated: showing lines 1-{DEFAULT_MAX_READ_LINES} of {total}. "
            "Use read_file_range to read the rest.]"
        )
        return {"stdout": f"{body}\n{footer}", "stderr": "", "exit_code": 0}
    return {"stdout": _number_lines(lines, 1), "stderr": "", "exit_code": 0}


def read_file_range(
    root: Path, path: str, start_line: int = 1, end_line: int | None = None
) -> dict[str, str | int]:
    if start_line < 1:
        return {"stdout": "", "stderr": "Error: 'start_line' must be >= 1.", "exit_code": 1}
    if end_line is None:
        end_line = start_line + DEFAULT_MAX_READ_LINES - 1
    if end_line < start_line:
        return {
            "stdout": "",
            "stderr": "Error: 'end_line' must be >= 'start_line'.",
            "exit_code": 1,
        }

    lines, error = _read_lines(root, path)
    if error is not None:
        return error

    total = len(lines)
    if start_line > total:
        return {
            "stdout": "",
            "stderr": f"Start line {start_line} is past the end of '{path}' ({total} lines).",
            "exit_code": 1,
        }

    last = min(end_line, total, start_line + DEFAULT_MAX_READ_LINES - 1)
    body = _number_lines(lines[start_line - 1 : last], start_line)
    footer = f"[lines {start_line}-{last} of {total}]"
    return {"stdout": f"{body}\n{footer}", "stderr": "", "exit_code": 0}


def replace_in_file(root: Path, path: str, old_str: str, new_str: str) -> dict[str, str | int]:
    if not old_str:
        return {"stdout": "", "stderr": "Error: 'old_str' must not be empty.", "exit_code": 1}
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    try:
        content = target.read_text()
    except UnicodeDecodeError:
        return {
            "stdout": "",
            "stderr": f"Cannot edit '{path}': not UTF-8 text (binary file).",
            "exit_code": 1,
        }
    occurrences = content.count(old_str)
    if occurrences == 0:
        return {
            "stdout": "",
            "stderr": f"No match for 'old_str' in '{path}'.",
            "exit_code": 1,
        }
    if occurrences > 1:
        return {
            "stdout": "",
            "stderr": (
                f"'old_str' is ambiguous in '{path}': found {occurrences} matches. "
                "Include surrounding lines to make it unique."
            ),
            "exit_code": 1,
        }
    target.write_text(content.replace(old_str, new_str))
    return {"stdout": f"Replaced 1 occurrence in '{path}'", "stderr": "", "exit_code": 0}


def write_files(root: Path, path: str, content: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"stdout": f"Wrote {len(content)} bytes to '{path}'", "stderr": "", "exit_code": 0}


def delete_file(root: Path, path: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    target.unlink()
    return {"stdout": f"Deleted '{path}'", "stderr": "", "exit_code": 0}


def search_files(root: Path, pattern: str) -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)
    if shutil.which("rg"):
        args = ["rg", "-n", pattern, "."]
    else:
        args = ["grep", "-rn", pattern, "."]
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "exit_code": 1}
