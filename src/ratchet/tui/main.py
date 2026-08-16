import asyncio
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog

from ratchet.agent.client import call_llm

DEFAULT_LOG_PATH = Path("log/ratchet.log")


class RatchetApp(App):
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
    ]

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        super().__init__()
        self.log_path = log_path

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="messages")
        yield Input(id="message_input")
        yield Footer()

    def on_mount(self) -> None:
        self._write_log("app launched")
        self.query_one("#message_input", Input).focus()

    def on_unmount(self) -> None:
        self._write_log("app stopped")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        if not text.strip():
            return
        self.query_one("#messages", RichLog).write(text)
        self._write_log(text)
        event.input.value = ""
        self._request_reply(text)

    @work
    async def _request_reply(self, text: str) -> None:
        result = await asyncio.to_thread(call_llm, [{"role": "user", "content": text}])
        self.query_one("#messages", RichLog).write(result["content"])
        self._write_log(result["content"])

    def _write_log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")

    def action_clear_log(self) -> None:
        self.query_one("#messages", RichLog).clear()
        self._write_log("log cleared")


def main() -> None:
    RatchetApp().run()


if __name__ == "__main__":
    main()
