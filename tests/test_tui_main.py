import json
import re

import pytest
from textual.events import Paste
from textual.widgets import Footer, Header, RichLog

from ratchet.agent import config as agent_config
from ratchet.tui import main as tui_main
from ratchet.tui.main import PromptInput, RatchetApp

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
    def fake_call_llm(messages, override_config=None, tools=None):
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
        input_widget = app.query_one("#message_input", PromptInput)
        assert app.focused is input_widget


async def test_typing_letter_q_does_not_quit(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        await pilot.press("q", "u", "i", "t")
        assert app.is_running
        assert input_widget.text == "quit"


async def test_submitted_message_echoes_to_display(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("\u203a hello there" in line for line in lines)


async def test_submitted_message_clears_input(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        assert input_widget.text == ""


async def test_submitted_message_written_to_log_file(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
    content = log_path.read_text()
    assert re.search(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} user: hello there$", content, re.MULTILINE
    )


async def test_multiple_messages_appended_in_order(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        for text in ["first", "second"]:
            input_widget.text = text
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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.text = ""
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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        richlog = app.query_one("#messages", RichLog)
        assert len(richlog.lines) > 0
        await pilot.press("ctrl+l")
        assert len(richlog.lines) == 0


async def test_agent_remembers_earlier_turns(tmp_path, monkeypatch):
    seen = []

    def fake_call_llm(messages, override_config=None, tools=None):
        seen.append(list(messages))
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "first"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        input_widget.text = "second"
        await pilot.press("enter")
        await app.workers.wait_for_complete()

    second_turn_messages = seen[1]
    assert {"role": "user", "content": "first"} in second_turn_messages
    assert {"role": "assistant", "content": "mock-reply"} in second_turn_messages
    assert second_turn_messages[-1] == {"role": "user", "content": "second"}


async def test_ctrl_l_resets_conversation_memory(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.press("ctrl+l")
        assert app.messages == [{"role": "system", "content": tui_main.SYSTEM_PROMPT}]


async def test_turn_saves_session_to_disk(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()

    assert json.loads(app.session_path.read_text()) == app.messages


async def test_app_loads_existing_session_on_mount(tmp_path):
    session_path = tmp_path / "session.json"
    saved = [
        {"role": "system", "content": tui_main.SYSTEM_PROMPT},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    session_path.write_text(json.dumps(saved))
    app = RatchetApp(log_path=tmp_path / "ratchet.log", session_path=session_path)

    async with app.run_test():
        assert app.messages == saved


async def test_ctrl_l_clears_session_file(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.session_path.exists()
        await pilot.press("ctrl+l")
        assert not app.session_path.exists()


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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "not yet submitted"
        await pilot.press("ctrl+l")
        assert input_widget.text == "not yet submitted"


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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("\u25c6 mock-reply" in line for line in lines)


async def test_agent_reply_written_to_log_file(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
    content = log_path.read_text()
    assert re.search(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} assistant: mock-reply$", content, re.MULTILINE
    )


async def test_offline_reply_content_is_still_displayed(tmp_path, monkeypatch):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {
            "content": "[LM Studio Offline] Could not connect to local server.",
            "model": "test-model",
            "status": "offline",
            "error": "connection refused",
        }

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("LM Studio Offline" in line for line in lines)


async def test_error_reply_content_is_still_displayed(tmp_path, monkeypatch):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {
            "content": "[API Error] boom",
            "model": "test-model",
            "status": "error",
            "error": "boom",
        }

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("[API Error] boom" in line for line in lines)


async def test_tool_call_shows_a_result_line_in_display(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a.txt").write_text("")
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "there is a.txt", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=sandbox)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "what files exist?"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert any("list_files" in line for line in lines)
        assert any("\u2713" in line and "a.txt (0 bytes)" in line for line in lines)


async def test_tool_call_logged_to_log_file(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("")
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "there is a.txt", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path, sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "what files exist?"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
    content = log_path.read_text()
    assert re.search(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} tool: list_files -> a\.txt \(0 bytes\)$",
        content,
        re.MULTILINE,
    )


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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "cat sample.txt"
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
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "cat /etc/passwd"
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

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(override_config)
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.press("enter")
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert calls[-1] == {"model": {"name": "anthropic/claude-sonnet-5"}}


async def test_user_message_logged_before_reply_worker_completes(tmp_path):
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello there"
        await pilot.press("enter")
        content = log_path.read_text()
        assert re.search(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} user: hello there$", content, re.MULTILINE
        )
        await app.workers.wait_for_complete()


async def test_multiline_paste_keeps_every_line(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        app.post_message(Paste("def foo():\n    return 1\n"))
        await pilot.pause()
        assert input_widget.text == "def foo():\n    return 1\n"


async def test_multiline_message_sent_to_agent_verbatim(tmp_path, monkeypatch):
    seen = []

    def fake_call_llm(messages, override_config=None, tools=None):
        seen.append(list(messages))
        return {"content": "mock-reply", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        app.post_message(Paste("first line\nsecond line"))
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert seen[0][-1]["content"] == "first line\nsecond line"
        assert input_widget.text == ""


async def test_newline_keys_insert_newline_instead_of_submitting(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        await pilot.press("a", "shift+enter", "b", "ctrl+j", "c")
        assert input_widget.text == "a\nb\nc"
        richlog = app.query_one("#messages", RichLog)
        lines = [strip.text for strip in richlog.lines]
        assert not any("user:" in line for line in lines)


def test_summarize_output_keeps_a_short_single_line():
    assert tui_main.summarize_output("Wrote 5 bytes to 'a.txt'") == "Wrote 5 bytes to 'a.txt'"


def test_summarize_output_collapses_multiple_lines_to_a_count():
    assert tui_main.summarize_output("one\ntwo\nthree") == "3 lines"


def test_summarize_output_truncates_a_long_single_line():
    summary = tui_main.summarize_output("x" * 300)
    assert len(summary) <= 100
    assert summary.endswith("…")


def test_summarize_output_handles_empty_output():
    assert tui_main.summarize_output("") == "(no output)"


def test_format_tool_line_marks_success_with_elapsed():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, index=1, name="list_files", output="a.txt",
        exit_code=0, elapsed=0.12
    )
    line = tui_main.format_tool_line(event)
    assert "✓" in line
    assert "0.1s" in line
    assert "a.txt" in line


def test_format_tool_line_marks_failure():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, name="read_files", output="File not found", exit_code=1
    )
    line = tui_main.format_tool_line(event)
    assert "✗" in line
    assert "✓" not in line


def test_format_tool_line_escapes_markup_in_output():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, name="read_files", output="1| [bold]hi", exit_code=0
    )
    assert "\\[bold]" in tui_main.format_tool_line(event)


def test_format_log_line_stays_plain_text():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, name="list_files", output="a.txt", exit_code=0
    )
    assert tui_main.format_log_line(event) == "tool: list_files -> a.txt"


async def test_status_bar_is_hidden_when_idle(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        assert not app.query_one("#status", tui_main.StatusBar).display


async def test_status_bar_shows_elapsed_while_working(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        bar = app.query_one("#status", tui_main.StatusBar)
        bar.start()
        assert bar.display
        assert bar.render_text().endswith("s")


async def test_status_bar_hides_when_stopped(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        bar = app.query_one("#status", tui_main.StatusBar)
        bar.start()
        bar.stop()
        assert not bar.display


async def test_status_bar_advance_changes_the_spinner_frame(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        bar = app.query_one("#status", tui_main.StatusBar)
        bar.start()
        first = bar.render_text()[0]
        bar.advance()
        assert bar.render_text()[0] != first


async def test_status_bar_is_hidden_again_after_a_turn(tmp_path, monkeypatch):
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert not app.query_one("#status", tui_main.StatusBar).display


async def test_prompt_and_reply_use_role_markers(tmp_path):
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        lines = [strip.text for strip in app.query_one("#messages", RichLog).lines]
        assert any(line.startswith("› hello") for line in lines)
        assert any("mock-reply" in line for line in lines)


async def test_error_reply_uses_the_error_marker(tmp_path, monkeypatch):
    def failing_call_llm(messages, override_config=None, tools=None):
        return {
            "content": "[API Error] boom",
            "model": "test-model",
            "status": "error",
            "error": "boom",
        }

    monkeypatch.setattr(tui_main, "call_llm", failing_call_llm)
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        lines = [strip.text for strip in app.query_one("#messages", RichLog).lines]
        assert any("[API Error] boom" in line for line in lines)


async def test_user_text_with_markup_is_not_interpreted(tmp_path):
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "read [bold]file"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        lines = [strip.text for strip in app.query_one("#messages", RichLog).lines]
        assert any("[bold]file" in line for line in lines)


def test_format_tool_args_uses_the_path_for_file_tools():
    assert tui_main.format_tool_args("read_files", {"path": "app/a.txt"}) == "app/a.txt"


def test_format_tool_args_uses_the_command_for_run_command():
    assert tui_main.format_tool_args("run_command", {"command": "tar -tf x.tar"}) == "tar -tf x.tar"


def test_format_tool_args_uses_the_query_for_search_web():
    assert tui_main.format_tool_args("search_web", {"query": "asyncio"}) == "asyncio"


def test_format_tool_args_shows_source_and_destination():
    args = tui_main.format_tool_args("copy_file", {"source": "a.txt", "destination": "b.txt"})
    assert args == "a.txt → b.txt"


def test_format_tool_args_is_empty_when_there_is_nothing_to_show():
    assert tui_main.format_tool_args("list_files", {}) == ""


def test_format_tool_args_truncates_a_long_value():
    args = tui_main.format_tool_args("run_command", {"command": "x" * 200})
    assert len(args) <= 80
    assert args.endswith("…")


def test_format_call_line_numbers_the_step_and_names_the_tool():
    event = tui_main.TurnEvent(
        phase="tool_start", step=1, index=3, name="run_command",
        arguments={"command": "tar -tf x.tar"}
    )
    line = tui_main.format_call_line(event)
    assert "3" in line
    assert "run_command" in line
    assert "tar -tf x.tar" in line


def test_format_call_line_escapes_markup_in_arguments():
    event = tui_main.TurnEvent(
        phase="tool_start", step=1, index=1, name="read_files", arguments={"path": "[a].txt"}
    )
    assert "\\[a]" in tui_main.format_call_line(event)


async def test_transcript_keeps_the_call_line_and_the_result(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a.txt").write_text("one\ntwo\nthree\nfour")
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
            }
        return {"content": "read it", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=sandbox)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "read a.txt"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        lines = [strip.text for strip in app.query_one("#messages", RichLog).lines]

    assert any("read_files" in line and "a.txt" in line for line in lines)
    assert any("✓" in line and "4 lines" in line for line in lines)
    assert not any("1| one" in line for line in lines)


async def test_call_line_is_logged_with_its_arguments(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a.txt").write_text("hello")
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
            }
        return {"content": "read it", "model": "test-model", "status": "success"}

    monkeypatch.setattr(tui_main, "call_llm", fake_call_llm)
    log_path = tmp_path / "ratchet.log"
    app = RatchetApp(log_path=log_path, sandbox_root=sandbox)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#message_input", PromptInput)
        input_widget.focus()
        input_widget.text = "read a.txt"
        await pilot.press("enter")
        await app.workers.wait_for_complete()

    content = log_path.read_text()
    assert "tool: read_files(a.txt)" in content
    assert "tool: read_files -> 1| hello" in content


def test_format_tool_line_colors_a_success_green():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, index=1, name="list_files", output="a.txt", exit_code=0
    )
    line = tui_main.format_tool_line(event)
    assert "[green]" in line
    assert "[red]" not in line


def test_format_tool_line_colors_the_whole_failure_red():
    event = tui_main.TurnEvent(
        phase="tool_done", step=1, index=1, name="read_files",
        output="File not found: 'a.txt'", exit_code=1
    )
    line = tui_main.format_tool_line(event)
    assert line.count("[red]") == 1
    assert line.rstrip().endswith("[/red]")
    assert "[green]" not in line


def test_format_reply_line_keeps_the_text_plain():
    line = tui_main.format_reply_line("all done")
    assert "all done" in line
    assert "bold" not in line
    assert "cyan" not in line


def test_format_reply_line_escapes_markup():
    assert "\\[bold]" in tui_main.format_reply_line("[bold]hi")


def test_format_error_line_is_red():
    line = tui_main.format_error_line("[API Error] boom")
    assert "[red]" in line
    assert "! " in line
