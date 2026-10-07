import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from ratchet.agent.client import call_llm
from ratchet.agent.config import load_config
from ratchet.agent.events import TurnEvent
from ratchet.agent.loop import DEFAULT_MAX_STEPS, run_agent_turn
from ratchet.agent.models import load_supported_models
from ratchet.agent.plan import PLAN_JSON_FILE, get_plan
from ratchet.agent.sessions import Session
from ratchet.agent.subagent import normalize_role
from ratchet.agent.tools import SYSTEM_PROMPT as SYSTEM_PROMPT
from ratchet.shell.executor import run_command
from ratchet.tui.tokens import TokenMeter

DEFAULT_LOG_PATH = Path("log/ratchet.log")
SUMMARY_WIDTH = 100
PREVIEW_LINES = 3
PREVIEW_WIDTH = 88
RESULT_UNITS = {
    "list_files": "entries",
    "file_search": "files",
    "search_files": "matches",
    "search_web": "results",
}
_SINGULAR = {"entries": "entry", "files": "file", "matches": "match", "results": "result",
             "lines": "line"}
ARGS_WIDTH = 80
NAME_WIDTH = 14
ARG_KEYS = ("command", "query", "url", "path", "pattern", "task", "name", "question")
SPAWN_TOOL = "spawn_subagent"
SPAWN_TOOLS = {SPAWN_TOOL, "spawn_parallel_subagent"}
PLAN_TOOLS = {"create_plan", "get_plan", "update_plan_step"}
PLAN_STATUS_COLOURS = {
    "pending": "dim",
    "in_progress": "yellow",
    "completed": "green",
    "skipped": "dim",
    "failed": "red",
}
ROLE_COLOURS = {
    "researcher": "cyan",
    "coder": "yellow",
    "tester": "magenta",
    "generalist": "white",
    "planner": "blue",
}


def summarize_output(output: str) -> str:
    text = output.strip()
    if not text:
        return "(no output)"
    lines = text.splitlines()
    if len(lines) > 1:
        return f"{len(lines)} lines"
    if len(text) > SUMMARY_WIDTH:
        return text[: SUMMARY_WIDTH - 1] + "\u2026"
    return text


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def format_tool_args(name: str, arguments: dict) -> str:
    source = arguments.get("source")
    destination = arguments.get("destination")
    if source and destination:
        return _truncate(f"{source} \u2192 {destination}", ARGS_WIDTH)
    if name in ("search_files", "file_search"):
        scope = str(arguments.get("path") or "").strip()
        pattern = f'"{arguments.get("pattern", "")}"'
        if scope and scope != ".":
            pattern += f" in {scope}"
        return _truncate(pattern, ARGS_WIDTH)
    if name == "read_file_range":
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line")
        span = f"{start}-{end}" if end else f"from {start}"
        return _truncate(f"{arguments.get('path', '')} {span}".strip(), ARGS_WIDTH)
    if name in ("write_files", "append_file"):
        size = len(str(arguments.get("content", "")))
        return _truncate(f"{arguments.get('path', '')} ({size} B)", ARGS_WIDTH)
    if name == "delete_file":
        path = str(arguments.get("path", ""))
        return _truncate(f"{path} (recursive)" if arguments.get("recursive") else path, ARGS_WIDTH)
    if name == "search_web":
        query = f'"{arguments.get("query", "")}"'
        count = arguments.get("max_results")
        return _truncate(f"{query} \u00d7{count}" if count else query, ARGS_WIDTH)
    for key in ARG_KEYS:
        value = arguments.get(key)
        if value:
            return _truncate(str(value), ARGS_WIDTH)
    return ""


def _count_results(name: str, text: str) -> int:
    if name == "search_web":
        return len([block for block in text.split("\n\n") if block.strip()])
    return len(text.splitlines())


def summarize_result(name: str, output: str) -> str:
    text = output.strip()
    if not text:
        return "(no output)"
    lines = text.splitlines()
    if len(lines) == 1:
        return _truncate(text, SUMMARY_WIDTH)
    unit = RESULT_UNITS.get(name, "lines")
    count = _count_results(name, text)
    if count == 1:
        unit = _SINGULAR.get(unit, unit)
    return f"{count} {unit}"


def output_preview_lines(output: str) -> list[str]:
    lines = output.strip().splitlines()
    if len(lines) <= 1:
        return []
    shown = [_truncate(line.rstrip(), PREVIEW_WIDTH) for line in lines[:PREVIEW_LINES]]
    if len(lines) > PREVIEW_LINES:
        shown.append(f"\u2026 {len(lines) - PREVIEW_LINES} more")
    return shown


def format_indent(event: TurnEvent) -> str:
    return "    " * event.depth


def format_call_line(event: TurnEvent) -> str:
    args = escape(format_tool_args(event.name, event.arguments))
    name = escape(event.name)
    body = f"{name:<{NAME_WIDTH}} {args}".rstrip() if args else name
    return f"{format_indent(event)}  [dim]{event.index}[/dim] [dim]\u25b8[/dim] {body}"


def format_call_start_line(event: TurnEvent) -> str:
    return f"{format_call_line(event)} [dim]\u2026 running[/dim]"


def format_tool_line(event: TurnEvent) -> str:
    summary = escape(summarize_result(event.name, event.output))
    indent = f"{format_indent(event)}      "
    if event.exit_code != 0:
        return f"{indent}[red]\u2717 {event.elapsed:.1f}s \u00b7 {summary}[/red]"
    return f"{indent}[green]\u2713[/green] [dim]{event.elapsed:.1f}s \u00b7[/dim] {summary}"


def format_lane_tag(event: TurnEvent) -> str:
    return f" #{event.lane}" if event.lane else ""


def role_colour(role: str) -> str:
    return ROLE_COLOURS.get(role, ROLE_COLOURS["generalist"])


def format_delegation_header(event: TurnEvent) -> str:
    role = normalize_role(str(event.arguments.get("role", "")))
    colour = role_colour(role)
    head = (
        f"{format_indent(event)}  [dim]{event.index}[/dim] [dim]\u25b8[/dim] "
        f"{escape(event.name):<{NAME_WIDTH}} [{colour}]{role}{format_lane_tag(event)}[/{colour}]"
        f" [dim]\u2026 running[/dim]"
    )
    task = _truncate(str(event.arguments.get("task", "")).strip(), ARGS_WIDTH)
    if not task:
        return head
    return f"{head}\n{format_indent(event)}      [dim]task[/dim]  {escape(task)}"


def split_subagent_output(output: str) -> tuple[str, list[str]]:
    lines = output.rstrip().splitlines()
    if not lines or not (lines[-1].startswith("[") and lines[-1].endswith("]")):
        return output.strip(), []
    parts = [p.strip() for p in lines[-1][1:-1].split("\u00b7")]
    shown = [p for p in parts[1:] if not p.startswith("via ")]
    return "\n".join(lines[:-1]).strip(), shown


def format_delegation_footer(event: TurnEvent) -> str:
    role = normalize_role(str(event.arguments.get("role", "")))
    colour = role_colour(role)
    summary, parts = split_subagent_output(event.output)
    indent = f"{format_indent(event)}  "
    if event.exit_code == 0:
        state = "[green]\u2713 done[/green]"
    else:
        state = "[red]\u2717 failed[/red]"
    meta = " \u00b7 ".join(parts) or f"{event.elapsed:.1f}s"
    lines = [f"{indent}{state} [{colour}]{format_lane_tag(event)} \u00b7 {escape(meta)}[/{colour}]"]
    if summary:
        lines.append(f"{indent}[dim]\u25c6[/dim] {escape(summary)}")
    return "\n".join(lines)


def format_tool_result_block(event: TurnEvent) -> str:
    lines = [format_tool_line(event)]
    indent = f"{format_indent(event)}        "
    lines += [f"{indent}[dim]{escape(line)}[/dim]" for line in output_preview_lines(event.output)]
    return "\n".join(lines)


def format_reply_line(reply: str) -> str:
    return f"[dim]\u25c6[/dim] {escape(reply)}"


def format_error_line(reply: str) -> str:
    return f"[red]! {escape(reply)}[/red]"


def format_prompt_panel(text: str) -> Panel:
    return Panel(f"[bold]› {escape(text)}[/bold]", border_style="cyan", expand=False)


def format_reply_panel(reply: str) -> Panel:
    return Panel(format_reply_line(reply), border_style="green", expand=False)


def format_error_panel(reply: str) -> Panel:
    return Panel(format_error_line(reply), border_style="red", expand=False)


def format_ack_panel(message: str) -> Panel:
    return Panel(f"[green]✓ {escape(message)}[/green]", border_style="green", expand=False)


def format_plan_panel(payload: dict) -> Panel:
    data = payload["plan"]
    progress = payload["progress"]
    body = Text()
    body.append(data["title"], style="bold")
    body.append(
        f"\nProgress: {progress['completed']}/{progress['total']} completed "
        f"({progress['percent']:.1f}%)\n",
        style="cyan",
    )
    if data["summary"]:
        body.append(f"\n{data['summary']}\n")
    for step in data["steps"]:
        body.append(f"\n{step['id']}. {step['description']} — ")
        body.append(step["status"], style=PLAN_STATUS_COLOURS[step["status"]])
        if step["target_files"]:
            body.append("\n  Files: " + ", ".join(step["target_files"]), style="dim")
        if step["notes"]:
            body.append(f"\n  Notes: {step['notes']}", style="dim")
    if data["risks"]:
        body.append("\n\nRisks", style="bold")
        for risk in data["risks"]:
            body.append(f"\n  • {risk}")
    complete = progress["completed"] == progress["total"]
    return Panel(body, title="Active plan", border_style="green" if complete else "cyan")


def plan_panel_for_event(event: TurnEvent, root: Path) -> Panel | None:
    if event.phase != "tool_done" or event.exit_code != 0 or event.name not in PLAN_TOOLS:
        return None
    result = get_plan(root)
    if result["exit_code"] != 0:
        return None
    return format_plan_panel(json.loads(result["stdout"]))


def _log_name(event: TurnEvent) -> str:
    return f"[{event.agent}] {event.name}" if event.agent else event.name


def format_log_line(event: TurnEvent) -> str:
    return f"tool: {_log_name(event)} -> {event.output}"


def format_call_log_line(event: TurnEvent) -> str:
    args = format_tool_args(event.name, event.arguments)
    return f"tool: {_log_name(event)}({args})"


def format_tool_panel(event: TurnEvent) -> Panel:
    rows = [format_call_line(event).strip(), format_tool_line(event).strip()]
    rows += [f"  [dim]{escape(line)}[/dim]" for line in output_preview_lines(event.output)]
    body = "\n".join(rows)
    border_style = "red" if event.exit_code != 0 else "green"
    title = f"{event.index} {escape(event.name)}"
    return Panel(body, title=title, border_style=border_style, expand=False)


def format_tool_start_panel(event: TurnEvent) -> Panel:
    body = f"{format_call_line(event).strip()} [dim]… running[/dim]"
    title = f"{event.index} {escape(event.name)}"
    return Panel(body, title=title, border_style="yellow", expand=False)


def _status_label(event: TurnEvent) -> str:
    label = f"{event.agent}{format_lane_tag(event)}"
    if event.phase == "thinking":
        return f"{label} thinking" if event.agent else "thinking"
    if event.agent:
        return f"{label} \u00b7 {event.name}"
    return f"running {event.name}"


class StatusBar(Static):
    FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._started = 0.0
        self._timer = None
        self._step = 0
        self._max_steps = 0
        self._label = ""

    def on_mount(self) -> None:
        self.display = False
        self._timer = self.set_interval(0.1, self.advance, pause=True)

    def start(self) -> None:
        self._started = time.monotonic()
        self._frame = 0
        self._step = 0
        self._max_steps = 0
        self._label = ""
        self.display = True
        self.update(self.render_text())
        if self._timer:
            self._timer.resume()

    def advance(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self.update(self.render_text())

    def set_step(self, step: int, max_steps: int, label: str = "") -> None:
        self._step = step
        self._max_steps = max_steps
        self._label = label
        self.update(self.render_text())

    def stop(self) -> None:
        if self._timer:
            self._timer.pause()
        self.display = False

    def render_text(self) -> str:
        text = f"{self.FRAMES[self._frame]} {time.monotonic() - self._started:.1f}s"
        if self._max_steps:
            text += f" · step {self._step}/{self._max_steps}"
            if self._label:
                text += f" · {self._label}"
        return text


class PromptInput(TextArea):
    NEWLINE_KEYS = ("shift+enter", "ctrl+j")

    class Submitted(Message):
        def __init__(self, prompt_input: "PromptInput", text: str) -> None:
            super().__init__()
            self.prompt_input = prompt_input
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class ModelPickerScreen(ModalScreen[str]):
    def __init__(self, models: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        yield OptionList(
            *(
                Option(config["name"], id=alias)
                for alias, config in self._models.items()
            )
        )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)


class QuestionScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    QuestionScreen {
        align: center middle;
        background: $background 70%;
    }
    QuestionScreen #question_dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    QuestionScreen Static {
        height: auto;
        margin-bottom: 1;
    }
    QuestionScreen OptionList {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    """

    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__()
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="question_dialog"):
            yield Static(self.question, markup=False)
            if self.options:
                yield OptionList(*(Option(escape(option)) for option in self.options))
            yield Input(placeholder="Enter a custom answer", id="question_answer")
            yield Static("Enter to answer · Tab to type a custom answer · Escape to cancel")

    def on_mount(self) -> None:
        self.query_one(OptionList if self.options else Input).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.options[event.option_index])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        answer = event.value.strip()
        if answer:
            self.dismiss(answer)
        else:
            self.notify("Please provide an answer.", severity="warning")

    def action_cancel(self) -> None:
        self.dismiss(None)


class RatchetApp(App):
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #conversation {
        height: 1fr;
    }

    #messages {
        width: 1fr;
        height: 1fr;
    }

    #active_plan {
        display: none;
        width: 40%;
        height: 1fr;
        border-left: solid $accent;
        padding: 0 1;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #tokens {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    #message_input {
        height: auto;
        max-height: 12;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("ctrl+r", "reset_memory", "Reset Memory"),
        Binding("ctrl+p", "pick_model", "Pick Model"),
        Binding("ctrl+b", "toggle_plan_mode", "Plan Mode"),
    ]

    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        mode: str = "chat",
        sandbox_root: Path | None = None,
        session_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.mode = mode
        self.sandbox_root = sandbox_root or (Path.cwd() / "sandbox")
        self.session_path = session_path or (log_path.parent / "session.json")
        self.selected_model: dict[str, str] | None = None
        self.plan_mode = False
        self._question_lock = asyncio.Lock()
        self._question_future: asyncio.Future | None = None
        self._parent_step = 0
        self.session = Session.open(self.session_path, self.sandbox_root)
        self.messages = self.session.messages

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="conversation"):
            yield RichLog(id="messages", markup=True)
            with VerticalScroll(id="active_plan"):
                yield Static(id="plan_content", markup=False)
        yield StatusBar(id="status")
        yield TokenMeter(id="tokens")
        yield PromptInput(id="message_input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "shell mode" if self.mode == "shell" else ""
        self._write_log("app launched")
        self._refresh_active_plan()
        meter = self.query_one("#tokens", TokenMeter)
        meter.display = self.mode != "shell"
        meter.reset_context(load_config().get("model", {}).get("context_window"))
        meter.session_tokens = self.session.usage.get("session_tokens")
        meter.context_tokens = self.session.usage.get("context_tokens")
        meter.context_window = self.session.usage.get("context_window", meter.context_window)
        meter.update(meter.render_tokens())
        if self.mode != "shell":
            model_name = load_config().get("model", {}).get("name", "unknown")
            message = f"model: {model_name}"
            self.query_one("#messages", RichLog).write(f"[dim]{escape(message)}[/dim]")
            self._write_log(message)
        self.query_one("#message_input", PromptInput).focus()

    def on_unmount(self) -> None:
        if self._question_future is not None and not self._question_future.done():
            self._question_future.set_result(None)
        self._write_log("app stopped")

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text
        if not text.strip():
            return
        self.query_one("#messages", RichLog).write(format_prompt_panel(text))
        self._write_log(f"user: {text}")
        event.prompt_input.text = ""
        self._request_reply(text)

    @work
    async def _request_reply(self, text: str) -> None:
        status = self.query_one("#status", StatusBar)
        status.start()
        try:
            if self.mode == "shell":
                result = await asyncio.to_thread(run_command, text, self.sandbox_root)
                output = (result["stdout"] + result["stderr"]).strip() or "(no output)"
                reply = f"{output} [exit {result['exit_code']}]"
                log_message = f"shell: {reply}"
            else:
                override_config = {"model": self.selected_model} if self.selected_model else None
                reply = await asyncio.to_thread(
                    run_agent_turn,
                    call_llm,
                    text,
                    self.sandbox_root,
                    override_config,
                    lambda event: self.call_from_thread(self._on_turn_event, event),
                    self.messages,
                    plan_mode=self.plan_mode,
                    plan_root=self.session.root,
                    ask_user=lambda question, options: self.call_from_thread(
                        self._ask_user_question, question, options
                    ),
                )
                log_message = f"assistant: {reply}"
                meter = self.query_one("#tokens", TokenMeter)
                self.session.usage = {
                    "session_tokens": meter.session_tokens,
                    "context_tokens": meter.context_tokens,
                    "context_window": meter.context_window,
                }
                await asyncio.to_thread(self.session.save)
        except Exception as e:
            reply = f"[Error] {type(e).__name__}: {e}"
            log_message = f"error: {traceback.format_exc().rstrip()}"
        finally:
            status.stop()

        panel = format_error_panel(reply) if reply.startswith("[") else format_reply_panel(reply)
        self.query_one("#messages", RichLog).write(panel)
        self._write_log(log_message)

    async def _ask_user_question(self, question: str, options: list[str]) -> str | None:
        async with self._question_lock:
            future = asyncio.get_running_loop().create_future()
            self._question_future = future

            def answered(answer: str | None) -> None:
                if not future.done():
                    future.set_result(answer)

            try:
                await self.push_screen(QuestionScreen(question, options), answered)
                return await future
            finally:
                self._question_future = None

    def _refresh_active_plan(self, card: Panel | None = None) -> None:
        view = self.query_one("#active_plan", VerticalScroll)
        content = self.query_one("#plan_content", Static)
        if self.mode == "shell":
            view.display = False
            return
        if card is None:
            result = get_plan(self.session.root)
            if result["exit_code"] == 0:
                card = format_plan_panel(json.loads(result["stdout"]))
            else:
                path = self.session.root / PLAN_JSON_FILE
                if path.exists() or path.is_symlink():
                    content.update(Panel(
                        Text(f"Cannot load active plan\n{result['stderr']}"), border_style="red"
                    ))
                    view.display = True
                else:
                    content.update(Text(
                        "No active plan\nSubmit a goal to create a structured plan."
                    ))
                    view.display = self.plan_mode
                return
        content.update(card)
        view.display = True

    def _on_turn_event(self, event: TurnEvent) -> None:
        if event.phase == "usage":
            self.query_one("#tokens", TokenMeter).record(event)
            return
        max_steps = load_config().get("agent", {}).get("max_steps", DEFAULT_MAX_STEPS)
        status = self.query_one("#status", StatusBar)
        messages_widget = self.query_one("#messages", RichLog)
        if event.depth == 0:
            self._parent_step = event.step
        if event.phase == "thinking":
            status.set_step(self._parent_step, max_steps, _status_label(event))
            return
        if event.phase == "tool_start":
            status.set_step(self._parent_step, max_steps, _status_label(event))
            if event.name in SPAWN_TOOLS and not event.depth:
                messages_widget.write(format_delegation_header(event))
            elif event.depth:
                messages_widget.write(format_call_start_line(event))
            else:
                messages_widget.write(format_tool_start_panel(event))
            self._write_log(format_call_log_line(event))
            return
        status.set_step(self._parent_step, max_steps)
        plan_card = plan_panel_for_event(event, self.session.root)
        if plan_card is not None:
            self._refresh_active_plan(plan_card)
            messages_widget.write(plan_card)
        elif event.name in SPAWN_TOOLS and not event.depth:
            messages_widget.write(format_delegation_footer(event))
        elif event.depth:
            messages_widget.write(format_tool_result_block(event))
        else:
            messages_widget.write(format_tool_panel(event))
        self._write_log(format_log_line(event))

    def _write_log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")

    def action_clear_log(self) -> None:
        messages_widget = self.query_one("#messages", RichLog)
        messages_widget.clear()
        messages_widget.write(format_ack_panel("Log cleared."))
        self._write_log("log cleared")

    def action_reset_memory(self) -> None:
        if self.query_one("#status", StatusBar).display:
            return
        messages_widget = self.query_one("#messages", RichLog)
        messages_widget.clear()
        self.session = Session.new(self.session_path)
        self.messages = self.session.messages
        self.query_one("#tokens", TokenMeter).reset_session()
        self._refresh_active_plan()
        messages_widget.write(format_ack_panel("Memory reset."))
        self._write_log("memory reset")

    def action_toggle_plan_mode(self) -> None:
        self.plan_mode = not self.plan_mode
        self._refresh_active_plan()
        self.sub_title = "plan mode" if self.plan_mode else ""
        message = "Plan mode on: read-only." if self.plan_mode else "Plan mode off."
        self.query_one("#messages", RichLog).write(format_ack_panel(message))
        self._write_log(message.lower())

    def action_pick_model(self) -> None:
        self._pick_model()

    @work
    async def _pick_model(self) -> None:
        models = load_supported_models()
        alias = await self.push_screen_wait(ModelPickerScreen(models))
        if not alias:
            return
        self.selected_model = models[alias]
        self.query_one("#tokens", TokenMeter).reset_context(
            self.selected_model.get("context_window")
        )
        message = f"model set to {self.selected_model['name']}"
        self.query_one("#messages", RichLog).write(escape(message))
        self._write_log(message)


def main(mode: str = "chat") -> None:
    RatchetApp(mode=mode).run()


if __name__ == "__main__":
    main()
