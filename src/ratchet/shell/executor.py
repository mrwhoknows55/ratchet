import shlex
import shutil
import subprocess
from pathlib import Path


def _resolve_path(root: Path, relative: str) -> Path | None:
    if relative.startswith("/") or ".." in Path(relative).parts:
        return None
    return root / relative


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
    target = _resolve_path(root, path)
    if target is None:
        return _access_denied(path)
    if not target.is_file():
        return {"stdout": "", "stderr": f"File not found: '{path}'", "exit_code": 1}
    return {"stdout": target.read_text(), "stderr": "", "exit_code": 0}


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
