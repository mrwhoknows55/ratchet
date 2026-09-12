import argparse
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from ratchet.agent.client import call_llm
from ratchet.agent.tools import TurnEvent, run_agent_turn
from ratchet.shell.executor import run_command
from ratchet.tui.main import (
    format_call_line,
    format_error_line,
    format_reply_line,
    format_tool_line,
)
from ratchet.tui.main import main as run_tui


def run_cli(prompt: str, mode: str = "chat", sandbox_root: Path | None = None) -> None:
    console = Console()
    root = sandbox_root or (Path.cwd() / "sandbox")
    console.print(f"[bold]› {escape(prompt)}[/bold]")

    if mode == "shell":
        result = run_command(prompt, root)
        output = (result["stdout"] + result["stderr"]).strip() or "(no output)"
        reply = f"{output} [exit {result['exit_code']}]"
    else:
        def on_event(event: TurnEvent) -> None:
            if event.phase == "thinking":
                return
            if event.phase == "tool_start":
                console.print(format_call_line(event))
            else:
                console.print(format_tool_line(event))

        reply = run_agent_turn(call_llm, prompt, root, None, on_event)

    console.print(format_error_line(reply) if reply.startswith("[") else format_reply_line(reply))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ratchet")
    parser.add_argument("args", nargs="*", metavar="[shell] [prompt ...]")
    args = list(parser.parse_args(argv).args)

    mode = "chat"
    if args and args[0] == "shell":
        mode = "shell"
        args.pop(0)

    prompt = " ".join(args).strip()
    if prompt:
        run_cli(prompt, mode)
    else:
        run_tui(mode=mode)


if __name__ == "__main__":
    main()
