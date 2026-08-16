from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


class RatchetApp(App):
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("hello from ratchet", id="body")
        yield Footer()


def main() -> None:
    RatchetApp().run()


if __name__ == "__main__":
    main()
