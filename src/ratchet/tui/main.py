import asyncio
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, RichLog
from textual.widgets.option_list import Option

from ratchet.agent.client import call_llm
from ratchet.agent.config import load_config
from ratchet.agent.models import load_supported_models
from ratchet.agent.tools import run_agent_turn
from ratchet.shell.executor import run_command

DEFAULT_LOG_PATH = Path("log/ratchet.log")


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

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("ctrl+p", "pick_model", "Pick Model"),
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
        self.selected_model: dict[str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="messages")
        yield Input(id="message_input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "shell mode" if self.mode == "shell" else ""
        self._write_log("app launched")
        if self.mode != "shell":
            model_name = load_config().get("model", {}).get("name", "unknown")
            message = f"model: {model_name}"
            self.query_one("#messages", RichLog).write(message)
            self._write_log(message)
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
            override_config = {"model": self.selected_model} if self.selected_model else None
            reply = await asyncio.to_thread(
                run_agent_turn, call_llm, text, self.sandbox_root, override_config
            )
            message = f"assistant: {reply}"
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
        self.query_one("#messages", RichLog).write(message)
        self._write_log(message)


def main(mode: str = "chat") -> None:
    RatchetApp(mode=mode).run()


if __name__ == "__main__":
    main()
