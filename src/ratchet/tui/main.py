import asyncio
import time
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.panel import Panel
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from ratchet.agent.client import call_llm
from ratchet.agent.config import load_config
from ratchet.agent.models import load_supported_models
from ratchet.agent.tools import (
    DEFAULT_MAX_STEPS,
    SYSTEM_PROMPT,
    TurnEvent,
    clear_session,
    load_session,
    run_agent_turn,
    save_session,
)
from ratchet.shell.executor import run_command

DEFAULT_LOG_PATH = Path("log/ratchet.log")
SUMMARY_WIDTH = 100
ARGS_WIDTH = 80
NAME_WIDTH = 14
ARG_KEYS = ("command", "query", "url", "path", "pattern")


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
    for key in ARG_KEYS:
        value = arguments.get(key)
        if value:
            return _truncate(str(value), ARGS_WIDTH)
    return ""


def format_call_line(event: TurnEvent) -> str:
    args = escape(format_tool_args(event.name, event.arguments))
    name = escape(event.name)
    body = f"{name:<{NAME_WIDTH}} {args}".rstrip() if args else name
    return f"  [dim]{event.index}[/dim] [dim]\u25b8[/dim] {body}"


def format_tool_line(event: TurnEvent) -> str:
    summary = escape(summarize_output(event.output))
    if event.exit_code != 0:
        return f"      [red]\u2717 {event.elapsed:.1f}s \u00b7 {summary}[/red]"
    return f"      [green]\u2713[/green] [dim]{event.elapsed:.1f}s \u00b7[/dim] {summary}"


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


def format_log_line(event: TurnEvent) -> str:
    return f"tool: {event.name} -> {event.output}"


def format_call_log_line(event: TurnEvent) -> str:
    args = format_tool_args(event.name, event.arguments)
    return f"tool: {event.name}({args})"


def format_tool_panel(event: TurnEvent) -> Panel:
    body = f"{format_call_line(event).strip()}\n{format_tool_line(event).strip()}"
    border_style = "red" if event.exit_code != 0 else "green"
    title = f"{event.index} {escape(event.name)}"
    return Panel(body, title=title, border_style=border_style, expand=False)


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


class RatchetApp(App):
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #messages {
        height: 1fr;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
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
        self.messages: list[dict] = load_session(self.session_path) or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="messages", markup=True)
        yield StatusBar(id="status")
        yield PromptInput(id="message_input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "shell mode" if self.mode == "shell" else ""
        self._write_log("app launched")
        if self.mode != "shell":
            model_name = load_config().get("model", {}).get("name", "unknown")
            message = f"model: {model_name}"
            self.query_one("#messages", RichLog).write(f"[dim]{escape(message)}[/dim]")
            self._write_log(message)
        self.query_one("#message_input", PromptInput).focus()

    def on_unmount(self) -> None:
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
                )
                log_message = f"assistant: {reply}"
                await asyncio.to_thread(save_session, self.messages, self.session_path)
        finally:
            status.stop()

        panel = format_error_panel(reply) if reply.startswith("[") else format_reply_panel(reply)
        self.query_one("#messages", RichLog).write(panel)
        self._write_log(log_message)

    def _on_turn_event(self, event: TurnEvent) -> None:
        max_steps = load_config().get("agent", {}).get("max_steps", DEFAULT_MAX_STEPS)
        status = self.query_one("#status", StatusBar)
        if event.phase == "thinking":
            status.set_step(event.step, max_steps, "thinking")
            return
        if event.phase == "tool_start":
            status.set_step(event.step, max_steps, f"running {event.name}")
            self._write_log(format_call_log_line(event))
            return
        status.set_step(event.step, max_steps)
        self.query_one("#messages", RichLog).write(format_tool_panel(event))
        self._write_log(format_log_line(event))

    def _write_log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")

    def action_clear_log(self) -> None:
        self.query_one("#messages", RichLog).clear()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        clear_session(self.session_path)
        self._write_log("log cleared")

    def action_reset_memory(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        clear_session(self.session_path)
        self._write_log("memory reset")

    def action_pick_model(self) -> None:
        self._pick_model()

    @work
    async def _pick_model(self) -> None:
        models = load_supported_models()
        alias = await self.push_screen_wait(ModelPickerScreen(models))
        if not alias:
            return
        self.selected_model = models[alias]
        message = f"model set to {self.selected_model['name']}"
        self.query_one("#messages", RichLog).write(escape(message))
        self._write_log(message)


def main(mode: str = "chat") -> None:
    RatchetApp(mode=mode).run()


if __name__ == "__main__":
    main()
