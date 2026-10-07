import argparse
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ratchet.agent.client import call_llm
from ratchet.agent.events import TurnEvent
from ratchet.agent.loop import run_agent_turn
from ratchet.agent.plan import EXECUTE_PLAN_PROMPT
from ratchet.agent.sessions import Session
from ratchet.agent.tools import clear_session
from ratchet.shell.executor import run_command
from ratchet.tui.main import (
    SPAWN_TOOLS,
    format_call_start_line,
    format_delegation_footer,
    format_delegation_header,
    format_error_panel,
    format_fallback_panel,
    format_prompt_panel,
    format_reply_panel,
    format_tool_panel,
    format_tool_result_block,
    plan_panel_for_event,
)
from ratchet.tui.main import main as run_tui

DEFAULT_SESSION_PATH = Path("log/session.json")


def ask_user_question(question: str, options: list[str]) -> str | None:
    console = Console()
    console.print(Panel(escape(question), title="Clarifying question", border_style="cyan"))
    for index, option in enumerate(options, 1):
        console.print(f"  {index}. {escape(option)}")
    while True:
        try:
            answer = console.input("Answer (number or custom text; Ctrl+C to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not answer:
            console.print("Please provide an answer.")
            continue
        if options and answer.isdecimal():
            choice = int(answer)
            if not 1 <= choice <= len(options):
                console.print("Choose a listed number or enter a custom answer.")
                continue
            return options[choice - 1]
        return answer


def run_cli(
    prompt: str,
    mode: str = "chat",
    sandbox_root: Path | None = None,
    session_path: Path | None = None,
    plan_mode: bool = False,
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
        session = Session.open(path, root)
        messages = session.messages

        def on_event(event: TurnEvent) -> None:
            if event.phase == "fallback":
                console.print(format_fallback_panel(event))
                return
            if event.phase == "usage":
                used = (event.prompt_tokens or 0) + (event.completion_tokens or 0)
                session.usage["session_tokens"] = (session.usage.get("session_tokens") or 0) + used
                if event.depth == 0:
                    session.usage.update(context_tokens=used, context_window=event.context_window)
                return
            if event.phase == "thinking":
                return
            if event.phase == "tool_start":
                if event.name in SPAWN_TOOLS and not event.depth:
                    console.print(format_delegation_header(event))
                elif event.depth:
                    console.print(format_call_start_line(event))
                else:
                    console.print(f"[dim]→ running {escape(event.name)}...[/dim]")
                return
            if event.name in SPAWN_TOOLS and not event.depth:
                console.print(format_delegation_footer(event))
                return
            plan_card = plan_panel_for_event(event, session.root)
            if plan_card is not None:
                console.print(plan_card)
                return
            console.print(
                format_tool_result_block(event) if event.depth else format_tool_panel(event)
            )

        reply = run_agent_turn(
            call_llm, prompt, root, None, on_event, messages, plan_mode=plan_mode,
            ask_user=ask_user_question,
            plan_root=session.root,
        )
        session.save()

    console.print(format_error_panel(reply) if reply.startswith("[") else format_reply_panel(reply))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ratchet")
    parser.add_argument(
        "--new", "--clear", "--reset", dest="reset_session", action="store_true"
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute-plan", action="store_true")
    parser.add_argument("args", nargs="*", metavar="[shell] [prompt ...]")
    parsed = parser.parse_args(argv)
    args = list(parsed.args)

    mode = "chat"
    if args and args[0] == "shell":
        mode = "shell"
        args.pop(0)

    prompt = " ".join(args).strip()
    if parsed.execute_plan:
        prompt = f"{EXECUTE_PLAN_PROMPT}\n\n{prompt}".strip()

    if parsed.reset_session:
        clear_session(DEFAULT_SESSION_PATH)
        if not prompt:
            Console().print("[bold yellow]Session history cleared.[/bold yellow]")
            return

    if prompt and parsed.plan:
        run_cli(prompt, mode, plan_mode=True)
    elif prompt:
        run_cli(prompt, mode)
    else:
        run_tui(mode=mode)


if __name__ == "__main__":
    main()
