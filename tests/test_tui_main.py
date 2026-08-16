import re

import pytest
from textual.widgets import Footer, Header, Input, RichLog

from ratchet.tui import main as tui_main
from ratchet.tui.main import RatchetApp


def make_app(tmp_path):
    return RatchetApp(log_path=tmp_path / "ratchet.log")


@pytest.fixture(autouse=True)
def _stub_call_llm(monkeypatch):
    def fake_call_llm(messages, override_config=None):
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    return fake_call_llm


async def test_app_has_header_and_footer(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        assert app.query_one(Header) is not None
        assert app.query_one(Footer) is not None


async def test_quit_binding_exits_app(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+q")
        assert not app.is_running


async def test_unbound_key_does_not_exit_app(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("x")
        assert app.is_running


async def test_input_is_focused_on_launch(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        input_widget = app.query_one("#message_input", Input)
        assert app.focused is input_widget


async def test_typing_letter_q_does_not_quit(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        await pilot.press("q", "u", "i", "t")
        assert app.is_running
        assert input_widget.value == "quit"


async def test_submitted_message_echoes_to_display(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("hello there" in line for line in lines)


async def test_submitted_message_clears_input(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        assert input_widget.value == ""


async def test_submitted_message_written_to_log_file(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
    content = log_path.read_text()
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} hello there$", content, re.MULTILINE)


async def test_multiple_messages_appended_in_order(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        for text in ["first", "second"]:
            input_widget.value = text
            await pilot.press("enter")
        await app.workers.wait_for_complete()
    lines = log_path.read_text().splitlines()
    user_lines = [line for line in lines if line.endswith("first") or line.endswith("second")]
    assert user_lines[0].endswith("first")
    assert user_lines[1].endswith("second")


async def test_empty_message_not_echoed_or_logged(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.value = ""
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        assert len(richlog.lines) == 0
    content = log_path.read_text()
    assert "app launched" in content
    assert "app stopped" in content


async def test_whitespace_only_message_not_echoed_or_logged(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.value = "   "
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        assert len(richlog.lines) == 0


async def test_log_directory_created_if_missing(tmp_path):
    log_path = tmp_path / "nested" / "log" / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test():
        pass
    assert log_path.exists()


async def test_app_launched_logged_on_mount(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test():
        content = log_path.read_text()
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} app launched$"
        assert re.search(pattern, content, re.MULTILINE)


async def test_app_stopped_logged_on_unmount(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test():
        pass
    content = log_path.read_text()
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} app stopped$", content, re.MULTILINE)


async def test_ctrl_l_clears_message_log(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        assert len(richlog.lines) > 0
        await pilot.press("ctrl+l")
        assert len(richlog.lines) == 0


async def test_ctrl_l_on_empty_log_does_not_error(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        richlog = app.query_one("#messages", RichLog)
        await pilot.press("ctrl+l")
        assert app.is_running
        assert len(richlog.lines) == 0


async def test_ctrl_l_does_not_clear_input_value(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "not yet submitted"
        await pilot.press("ctrl+l")
        assert input_widget.value == "not yet submitted"


async def test_ctrl_l_logged_to_file(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+l")
    content = log_path.read_text()
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} log cleared$", content, re.MULTILINE)


async def test_agent_reply_written_to_display(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("mock-reply" in line for line in lines)


async def test_agent_reply_written_to_log_file(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
    content = log_path.read_text()
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} mock-reply$", content, re.MULTILINE)


async def test_offline_reply_content_is_still_displayed(tmp_path, monkeypatch):
    def fake_call_llm(messages, override_config=None):
        return {
            "content": "[LM Studio Offline] Could not connect to local server.",
            "model": "test-model",
            "status": "offline",
            "error": "connection refused",
        }

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("LM Studio Offline" in line for line in lines)


async def test_error_reply_content_is_still_displayed(tmp_path, monkeypatch):
    def fake_call_llm(messages, override_config=None):
        return {
            "content": "[API Error] boom",
            "model": "test-model",
            "status": "error",
            "error": "boom",
        }

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("[API Error] boom" in line for line in lines)


async def test_user_message_logged_before_reply_worker_completes(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        content = log_path.read_text()
        assert re.search(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} hello there$", content, re.MULTILINE
        )
        await app.workers.wait_for_complete()
