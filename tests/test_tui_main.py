from textual.widgets import Footer, Header, Static

from ratchet.tui.main import RatchetApp


async def test_app_shows_body_text():
    app = RatchetApp()
    async with app.run_test():
        body = app.query_one("#body", Static)
        assert body.content == "hello from ratchet"


async def test_app_has_header_and_footer():
    app = RatchetApp()
    async with app.run_test():
        assert app.query_one(Header) is not None
        assert app.query_one(Footer) is not None


async def test_quit_binding_exits_app():
    app = RatchetApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
        assert not app.is_running


async def test_unbound_key_does_not_exit_app():
    app = RatchetApp()
    async with app.run_test() as pilot:
        await pilot.press("x")
        assert app.is_running
