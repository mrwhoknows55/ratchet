import shlex
import subprocess
from pathlib import Path


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
