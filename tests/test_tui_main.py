import re

import pytest
from textual.widgets import Footer, Header, Input, RichLog

from ratchet.agent import config as agent_config
from ratchet.tui import main as tui_main
from ratchet.tui.main import RatchetApp

TEST_CONFIG_TOML = """
[model]
provider = "openrouter"
base_url = "https://openrouter.ai/api/v1"
name = "anthropic/claude-sonnet-5"
api_key = "not-needed"
timeout = 20

[models.claude]
name = "anthropic/claude-sonnet-5"

[models.lmstudio]
name = "liquid/lfm2.5-1.2b"
base_url = "http://localhost:1234/v1"
"""


def make_app(tmp_path):
    return RatchetApp(log_path=tmp_path / "ratchet.log")


@pytest.fixture(autouse=True)
def _stub_call_llm(monkeypatch):
    def fake_call_llm(messages, override_config=None):
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    return fake_call_llm


@pytest.fixture(autouse=True)
def _isolate_config_file(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(TEST_CONFIG_TOML)
    monkeypatch.setattr(agent_config, "CONFIG_FILE", config_file)


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
        assert any("user: hello there" in line for line in lines)


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
    assert re.search(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} user: hello there$", content, re.MULTILINE
    )


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
        richlog = app.query_one("#messages", RichLog)
        lines_before = len(richlog.lines)
        input_widget = app.query_one("#message_input", Input)
        input_widget.value = ""
        await pilot.press("enter")
        assert len(richlog.lines) == lines_before
    content = log_path.read_text()
    assert "app launched" in content
    assert "app stopped" in content


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


async def test_active_model_name_printed_on_launch(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("model: anthropic/claude-sonnet-5" in line for line in lines)


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
        assert any("assistant: mock-reply" in line for line in lines)


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
    assert re.search(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} assistant: mock-reply$", content, re.MULTILINE
    )


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


async def test_shell_mode_defaults_sandbox_root_to_cwd_sandbox_subdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = RatchetApp(log_path=tmp_path / "ratchet.log", mode="shell")
    assert app.sandbox_root == tmp_path / "sandbox"


async def test_shell_mode_executes_command_and_displays_output(tmp_path):
    (tmp_path / "sample.txt").write_text("hello sandbox")
    app = RatchetApp(
        log_path=tmp_path / "ratchet.log", mode="shell", sandbox_root=tmp_path
    )
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "cat sample.txt"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("hello sandbox" in line and "exit 0" in line for line in lines)


async def test_shell_mode_denies_path_outside_sandbox(tmp_path):
    app = RatchetApp(
        log_path=tmp_path / "ratchet.log", mode="shell", sandbox_root=tmp_path
    )
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "cat /etc/passwd"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("Access Denied" in line for line in lines)


async def test_model_picker_options_show_full_model_names(tmp_path):
    from textual.widgets import OptionList

    from ratchet.tui.main import ModelPickerScreen

    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ModelPickerScreen)
        option_list = screen.query_one(OptionList)
        prompts = {
            option_list.get_option_at_index(i).prompt
            for i in range(option_list.option_count)
        }
        assert prompts == {"anthropic/claude-sonnet-5", "liquid/lfm2.5-1.2b"}


async def test_ctrl_p_pick_model_sets_selected_model_and_confirms(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press("enter")
        assert app.selected_model == {"name": "anthropic/claude-sonnet-5"}
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("model set to anthropic/claude-sonnet-5" in line for line in lines)


async def test_chat_message_after_pick_uses_selected_model_override(tmp_path, monkeypatch):
    calls = []

    def fake_call_llm(messages, override_config=None):
        calls.append(override_config)
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.press("enter")
        input_widget = app.query_one("#message_input", Input)
        input_widget.focus()
        input_widget.value = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert calls[-1] == {"model": {"name": "anthropic/claude-sonnet-5"}}


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
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} user: hello there$", content, re.MULTILINE
        )
        await app.workers.wait_for_complete()
