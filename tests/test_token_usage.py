import json

import pytest

from ratchet.agent.events import TurnEvent
from ratchet.agent.loop import run_turn
from ratchet.tui import main as tui_main
from ratchet.tui.tokens import TokenMeter


@pytest.fixture
async def meter(tmp_path):
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        yield app.query_one("#tokens", TokenMeter)


def test_usage_events_keep_request_counts_separate_from_turn_totals(tmp_path):
    replies = iter([
        {"status": "success", "content": "", "usage": {
            "prompt_tokens": 100, "completion_tokens": 10,
        }, "context_window": 4096, "tool_calls": [{
            "id": "list", "type": "function", "function": {
                "name": "list_files", "arguments": json.dumps({}),
            },
        }]},
        {"status": "success", "content": "Done", "usage": {
            "prompt_tokens": 180, "completion_tokens": 20,
        }, "context_window": 4096},
    ])
    events = []
    result = run_turn(lambda *args, **kwargs: next(replies), "inspect", tmp_path,
                      on_event=events.append)
    usage = [event for event in events if event.phase == "usage"]
    assert [(event.prompt_tokens, event.completion_tokens, event.context_window)
            for event in usage] == [(100, 10, 4096), (180, 20, 4096)]
    assert result.prompt_tokens == 280
    assert result.completion_tokens == 30


@pytest.mark.parametrize("used,colour", [(250, "cyan"), (800, "yellow"), (1100, "red")])
async def test_meter_shows_actual_context_ratio_and_session_total(meter, used, colour):
    meter.record(TurnEvent(phase="usage", step=1, prompt_tokens=used - 10,
                           completion_tokens=10, context_window=1000))
    rendered = meter.render_tokens()
    assert f"{used:,} / 1,000" in rendered.plain
    assert f"{used / 10:.0f}%" in rendered.plain
    assert f"session {used:,} tok" in rendered.plain
    assert any(span.style == colour for span in rendered.spans)


async def test_subagent_usage_adds_to_session_without_replacing_parent_context(meter):
    meter.record(TurnEvent(phase="usage", step=1, prompt_tokens=100,
                           completion_tokens=20, context_window=1000))
    meter.record(TurnEvent(phase="usage", step=1, prompt_tokens=500,
                           completion_tokens=50, context_window=2000, depth=1))
    assert "120 / 1,000" in meter.render_tokens().plain
    assert "session 670 tok" in meter.render_tokens().plain
    meter.record(TurnEvent(phase="usage", step=2, prompt_tokens=200,
                           completion_tokens=30, context_window=1000))
    assert "230 / 1,000" in meter.render_tokens().plain
    assert "session 900 tok" in meter.render_tokens().plain


async def test_unknown_window_and_model_switch_do_not_show_a_false_percentage(meter):
    assert "— / unknown" in meter.render_tokens().plain
    meter.record(TurnEvent(phase="usage", step=1, prompt_tokens=100, completion_tokens=20))
    assert "120 / unknown" in meter.render_tokens().plain
    assert "%" not in meter.render_tokens().plain
    meter.reset_context(4096)
    assert "— / 4,096" in meter.render_tokens().plain
    assert "session 120 tok" in meter.render_tokens().plain


async def test_live_meter_persists_after_reply_and_resets_with_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(tui_main, "call_llm", lambda *args, **kwargs: {
        "status": "success", "content": "Done", "usage": {
            "prompt_tokens": 1500, "completion_tokens": 100,
        }, "context_window": 8000,
    })
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        app.query_one("#message_input", tui_main.PromptInput).text = "hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        meter = app.query_one("#tokens", TokenMeter)
        assert meter.display
        assert "1,600 / 8,000" in meter.render_tokens().plain
        assert "20%" in meter.render_tokens().plain
        assert "session 1,600 tok" in meter.render_tokens().plain
        await pilot.press("ctrl+r")
        assert "— / 8,000" in meter.render_tokens().plain
        assert "session —" in meter.render_tokens().plain


async def test_shell_mode_hides_the_token_meter(tmp_path):
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path,
                              mode="shell")
    async with app.run_test():
        assert not app.query_one("#tokens", TokenMeter).display
