import argparse
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from ratchet.agent.client import call_llm
from ratchet.agent.events import TurnEvent
from ratchet.agent.loop import run_agent_turn
from ratchet.agent.tools import (
    SYSTEM_PROMPT,
    clear_session,
    load_session,
    save_session,
)
from ratchet.shell.executor import run_command
from ratchet.tui.main import (
    format_call_line,
    format_error_panel,
    format_prompt_panel,
    format_reply_panel,
    format_tool_line,
    format_tool_panel,
)
from ratchet.tui.main import main as run_tui

DEFAULT_SESSION_PATH = Path("log/session.json")


def run_cli(
    prompt: str,
    mode: str = "chat",
    sandbox_root: Path | None = None,
    session_path: Path | None = None,
) -> None:
    console = Console()
    root = sandbox_root or (Path.cwd() / "sandbox")
    console.print(format_prompt_panel(prompt))

    if mode == "shell":
        result = run_command(prompt, root)
        output = (result["stdout"] + result["stderr"]).strip() or "(no output)"
        reply = f"{output} [exit {result['exit_code']}]"
    else:
        path = session_path or DEFAULT_SESSION_PATH
        messages = load_session(path) or [{"role": "system", "content": SYSTEM_PROMPT}]

        def on_event(event: TurnEvent) -> None:
            if event.phase == "thinking":
                return
            if event.phase == "tool_start":
                if event.depth:
                    console.print(format_call_line(event))
                else:
                    console.print(f"[dim]→ running {escape(event.name)}...[/dim]")
                return
            console.print(format_tool_line(event) if event.depth else format_tool_panel(event))

        reply = run_agent_turn(call_llm, prompt, root, None, on_event, messages)
        save_session(messages, path)

    console.print(format_error_panel(reply) if reply.startswith("[") else format_reply_panel(reply))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ratchet")
    parser.add_argument(
        "--new", "--clear", "--reset", dest="reset_session", action="store_true"
    )
    parser.add_argument("args", nargs="*", metavar="[shell] [prompt ...]")
    parsed = parser.parse_args(argv)
    args = list(parsed.args)

    mode = "chat"
    if args and args[0] == "shell":
        mode = "shell"
        args.pop(0)

    prompt = " ".join(args).strip()

    if parsed.reset_session:
        clear_session(DEFAULT_SESSION_PATH)
        if not prompt:
            Console().print("[bold yellow]Session history cleared.[/bold yellow]")
            return

    if prompt:
        run_cli(prompt, mode)
    else:
        run_tui(mode=mode)


if __name__ == "__main__":
    main()
