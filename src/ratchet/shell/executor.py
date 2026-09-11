import hashlib
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

BACKUP_DIR = ".backups"
DEFAULT_COMMAND_TIMEOUT = 10
MAX_COMMAND_TIMEOUT = 600
DEFAULT_MAX_READ_LINES = 200
DEFAULT_MAX_SEARCH_RESULTS = 100


def _resolve_path(
    root: Path,
    relative: str,
    *,
    must_exist: bool = False,
    allow_dir: bool = False,
    forbid_root: bool = False,
) -> tuple[Path | None, dict[str, str | int] | None]:
    if relative.startswith("/") or ".." in Path(relative).parts:
        return None, _access_denied(relative)
    target = root / relative
    if forbid_root and target.resolve() == root.resolve():
        return None, {
            "stdout": "",
            "stderr": f"Error: '{relative}' is the sandbox root; refusing to operate on it.",
            "exit_code": 1,
        }
    if must_exist and not target.exists():
        return None, {"stdout": "", "stderr": f"File not found: '{relative}'", "exit_code": 1}
    if not allow_dir and target.is_dir():
        return None, {
            "stdout": "",
            "stderr": f"Error: '{relative}' is a directory, not a file.",
            "exit_code": 1,
        }
    return target, None


def _number_lines(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(lines, start))


def _read_lines(root: Path, path: str) -> tuple[list[str] | None, dict[str, str | int] | None]:
    target, error = _resolve_path(root, path)
    if error is not None:
        return None, error
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


def _backup_tree(root: Path, directory: Path) -> None:
    for entry in directory.rglob("*"):
        if entry.is_file() and BACKUP_DIR not in entry.relative_to(root).parts:
            backup_file(root, entry)


def _is_inside(parent: Path, candidate: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def _access_denied(path: str) -> dict[str, str | int]:
    return {
        "stdout": "",
        "stderr": (
            f"Access Denied: Path traversal or absolute path '{path}' forbidden outside sandbox."
        ),
        "exit_code": 1,
    }


def run_command(command: str, root: Path, timeout: int | None = None) -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)

    command_str = command.strip()
    if not command_str:
        return {"stdout": "", "stderr": "Error: Empty command provided.", "exit_code": 1}

    for token in command_str.split():
        if token.startswith("/") or ".." in token:
            return _access_denied(token)

    limit = min(timeout or DEFAULT_COMMAND_TIMEOUT, MAX_COMMAND_TIMEOUT)
    args = shlex.split(command_str)
    try:
        result = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=limit,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": (
                f"Command timed out after {limit}s. Pass a larger 'timeout' "
                f"(up to {MAX_COMMAND_TIMEOUT}s) for slow work like downloads."
            ),
            "exit_code": 1,
        }
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: '{args[0]}'", "exit_code": 127}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "exit_code": 1}


def list_files(root: Path, path: str = ".") -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)
    target, error = _resolve_path(root, path, allow_dir=True)
    if error is not None:
        return error
    if not target.is_dir():
        return {"stdout": "", "stderr": f"Directory not found: '{path}'", "exit_code": 1}

    at_root = target.resolve() == root.resolve()
    entries = []
    for entry in sorted(target.iterdir(), key=lambda item: item.name):
        if at_root and entry.name == BACKUP_DIR:
            continue
        if entry.is_dir():
            entries.append(f"{entry.name}/")
        else:
            entries.append(f"{entry.name} ({entry.stat().st_size} bytes)")
    return {"stdout": "\n".join(entries), "stderr": "", "exit_code": 0}


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
    target, error = _resolve_path(root, path)
    if error is not None:
        return error
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
    target, error = _resolve_path(root, path)
    if error is not None:
        return error
    backup_file(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"stdout": f"Wrote {len(content)} bytes to '{path}'", "stderr": "", "exit_code": 0}


def append_file(root: Path, path: str, content: str) -> dict[str, str | int]:
    target, error = _resolve_path(root, path)
    if error is not None:
        return error
    backup_file(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        handle.write(content)
    return {
        "stdout": f"Appended {len(content)} bytes to '{path}'",
        "stderr": "",
        "exit_code": 0,
    }


def delete_file(root: Path, path: str, recursive: bool = False) -> dict[str, str | int]:
    target, error = _resolve_path(root, path, allow_dir=True, forbid_root=True)
    if error is not None:
        return error
    if target.is_dir():
        if not recursive:
            return {
                "stdout": "",
                "stderr": (
                    f"Error: '{path}' is a directory. "
                    "Pass recursive=true to delete it and everything inside it."
                ),
                "exit_code": 1,
            }
        _backup_tree(root, target)
        shutil.rmtree(target)
        return {"stdout": f"Deleted directory '{path}'", "stderr": "", "exit_code": 0}
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    backup_file(root, target)
    target.unlink()
    return {"stdout": f"Deleted '{path}'", "stderr": "", "exit_code": 0}


def copy_file(root: Path, source: str, destination: str) -> dict[str, str | int]:
    src, error = _resolve_path(root, source, must_exist=True, allow_dir=True, forbid_root=True)
    if error is not None:
        return error
    dst, error = _resolve_path(root, destination, allow_dir=True, forbid_root=True)
    if error is not None:
        return error

    if src.is_dir():
        if _is_inside(src, dst):
            return {
                "stdout": "",
                "stderr": f"Error: cannot copy '{source}' into itself.",
                "exit_code": 1,
            }
        if dst.exists():
            _backup_tree(root, dst)
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return {
            "stdout": f"Copied directory '{source}' to '{destination}'",
            "stderr": "",
            "exit_code": 0,
        }

    backup_file(root, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"stdout": f"Copied '{source}' to '{destination}'", "stderr": "", "exit_code": 0}


def move_file(root: Path, source: str, destination: str) -> dict[str, str | int]:
    src, error = _resolve_path(root, source, must_exist=True, allow_dir=True, forbid_root=True)
    if error is not None:
        return error
    dst, error = _resolve_path(root, destination, allow_dir=True, forbid_root=True)
    if error is not None:
        return error

    if src.is_dir():
        if _is_inside(src, dst):
            return {
                "stdout": "",
                "stderr": f"Error: cannot move '{source}' into itself.",
                "exit_code": 1,
            }
        _backup_tree(root, src)
        if dst.exists():
            _backup_tree(root, dst)
            shutil.rmtree(dst)
    else:
        backup_file(root, src)
        backup_file(root, dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"stdout": f"Moved '{source}' to '{destination}'", "stderr": "", "exit_code": 0}


def get_file_info(root: Path, path: str) -> dict[str, str | int]:
    target, error = _resolve_path(root, path)
    if error is not None:
        return error
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}

    data = target.read_bytes()
    try:
        lines = len(data.decode().splitlines())
    except UnicodeDecodeError:
        lines = -1
    modified = datetime.fromtimestamp(target.stat().st_mtime).isoformat(timespec="seconds")
    report = "\n".join(
        [
            f"path: {path}",
            f"size: {len(data)}",
            f"lines: {lines}",
            f"modified: {modified}",
            f"sha256: {hashlib.sha256(data).hexdigest()}",
        ]
    )
    return {"stdout": report, "stderr": "", "exit_code": 0}


def search_files(root: Path, pattern: str, path: str = ".") -> dict[str, str | int]:
    root.mkdir(parents=True, exist_ok=True)
    target, error = _resolve_path(root, path, allow_dir=True)
    if error is not None:
        return error
    if not target.is_dir():
        return {"stdout": "", "stderr": f"Directory not found: '{path}'", "exit_code": 1}
    if shutil.which("rg"):
        args = ["rg", "-n", "--glob", f"!{BACKUP_DIR}", pattern, path]
    else:
        args = ["grep", "-rn", f"--exclude-dir={BACKUP_DIR}", pattern, path]
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

    target, error = _resolve_path(root, path, allow_dir=True)
    if error is not None:
        return error
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
    target, error = _resolve_path(root, path)
    if error is not None:
        return error
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
