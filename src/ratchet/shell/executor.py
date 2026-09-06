import shlex
import shutil
import subprocess
from pathlib import Path

BACKUP_DIR = ".backups"
DEFAULT_MAX_READ_LINES = 200
DEFAULT_MAX_SEARCH_RESULTS = 100


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


def _backup_path(root: Path, target: Path) -> Path:
    return root / BACKUP_DIR / target.relative_to(root)


def backup_file(root: Path, target: Path) -> None:
    if not target.is_file():
        return
    snapshot = _backup_path(root, target)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, snapshot)


def restore_backup(root: Path, target: Path) -> bool:
    snapshot = _backup_path(root, target)
    if not snapshot.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, target)
    return True


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
            return _access_denied(token)

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
    names = sorted(entry.name for entry in root.iterdir() if entry.name != BACKUP_DIR)
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
    backup_file(root, target)
    target.write_text(content.replace(old_str, new_str))
    return {"stdout": f"Replaced 1 occurrence in '{path}'", "stderr": "", "exit_code": 0}


def write_files(root: Path, path: str, content: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    backup_file(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"stdout": f"Wrote {len(content)} bytes to '{path}'", "stderr": "", "exit_code": 0}


def append_file(root: Path, path: str, content: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    backup_file(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        handle.write(content)
    return {
        "stdout": f"Appended {len(content)} bytes to '{path}'",
        "stderr": "",
        "exit_code": 0,
    }


def delete_file(root: Path, path: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    backup_file(root, target)
    target.unlink()
    return {"stdout": f"Deleted '{path}'", "stderr": "", "exit_code": 0}


def search_files(root: Path, pattern: str) -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)
    if shutil.which("rg"):
        args = ["rg", "-n", "--glob", f"!{BACKUP_DIR}", pattern, "."]
    else:
        args = ["grep", "-rn", f"--exclude-dir={BACKUP_DIR}", pattern, "."]
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "exit_code": 1}


def file_search(root: Path, pattern: str, path: str = ".") -> dict[str, str | int]:
    if not pattern:
        return {"stdout": "", "stderr": "Error: 'pattern' must not be empty.", "exit_code": 1}
    if "/" in pattern or ".." in pattern:
        return {
            "stdout": "",
            "stderr": (
                f"Error: 'pattern' must be a bare name glob without '/' or '..': '{pattern}'. "
                "Use 'path' to scope the search to a subdirectory."
            ),
            "exit_code": 1,
        }

    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not target.is_dir():
        return {"stdout": "", "stderr": f"Directory not found: '{path}'", "exit_code": 1}

    matches = sorted(
        str(match.relative_to(root))
        for match in target.rglob(pattern)
        if BACKUP_DIR not in match.relative_to(root).parts
    )
    if not matches:
        return {"stdout": f"No files matching '{pattern}'", "stderr": "", "exit_code": 0}

    total = len(matches)
    if total > DEFAULT_MAX_SEARCH_RESULTS:
        body = "\n".join(matches[:DEFAULT_MAX_SEARCH_RESULTS])
        footer = f"[truncated: showing {DEFAULT_MAX_SEARCH_RESULTS} of {total} matches.]"
        return {"stdout": f"{body}\n{footer}", "stderr": "", "exit_code": 0}
    return {"stdout": "\n".join(matches), "stderr": "", "exit_code": 0}


def rollback_file(root: Path, path: str) -> dict[str, str | int]:
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not restore_backup(root, target):
        return {
            "stdout": "",
            "stderr": f"No snapshot to restore for '{path}'.",
            "exit_code": 1,
        }
    return {
        "stdout": f"Restored '{path}' from its last snapshot",
        "stderr": "",
        "exit_code": 0,
    }
