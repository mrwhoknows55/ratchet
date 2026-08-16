import asyncio
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog

from ratchet.agent.client import call_llm
from ratchet.shell.executor import run_command

DEFAULT_LOG_PATH = Path("log/ratchet.log")


class RatchetApp(App):
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
    ]

    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        mode: str = "chat",
        sandbox_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.mode = mode
        self.sandbox_root = sandbox_root or (Path.cwd() / "sandbox")

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="messages")
        yield Input(id="message_input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "shell mode" if self.mode == "shell" else ""
        self._write_log("app launched")
        self.query_one("#message_input", Input).focus()

    def on_unmount(self) -> None:
        self._write_log("app stopped")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        if not text.strip():
            return
        message = f"user: {text}"
        self.query_one("#messages", RichLog).write(message)
        self._write_log(message)
        event.input.value = ""
        self._request_reply(text)

    @work
    async def _request_reply(self, text: str) -> None:
        if self.mode == "shell":
            result = await asyncio.to_thread(run_command, text, self.sandbox_root)
            output = (result["stdout"] + result["stderr"]).strip() or "(no output)"
            message = f"shell: {output} [exit {result['exit_code']}]"
        else:
            result = await asyncio.to_thread(call_llm, [{"role": "user", "content": text}])
            message = f"assistant: {result['content']}"
        self.query_one("#messages", RichLog).write(message)
        self._write_log(message)

    def _write_log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")

    def action_clear_log(self) -> None:
        self.query_one("#messages", RichLog).clear()
        self._write_log("log cleared")


def main(mode: str = "chat") -> None:
    RatchetApp(mode=mode).run()


if __name__ == "__main__":
    main()
