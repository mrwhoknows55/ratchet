import json

import pytest
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static

from ratchet import cli
from ratchet.agent.events import TurnEvent
from ratchet.agent.plan import create_plan, get_plan
from ratchet.agent.tools import execute_tool_result
from ratchet.tui import main as tui_main


def test_plan_card_shows_progress_statuses_and_literal_plan_text(tmp_path):
    create_plan(tmp_path, "[red]Math[/red]", [
        {"id": "divide", "description": "Safe division", "status": "completed",
         "target_files": ["math_ops.py"], "notes": "Verified zero denominator"},
        {"description": "Power", "status": "in_progress"},
        {"description": "Cache", "status": "pending"},
        {"description": "Benchmark", "status": "skipped"},
        {"description": "Integration", "status": "failed"},
    ], summary="Small utilities", risks=["Cache invalidation"])
    payload = json.loads(get_plan(tmp_path)["stdout"])
    card = tui_main.format_plan_panel(payload)
    body = card.renderable.plain
    assert "[red]Math[/red]" in body
    assert "1/5 completed (20.0%)" in body
    for status in ("completed", "in_progress", "pending", "skipped", "failed"):
        assert status in body
    assert "divide. Safe division" in body
    assert "math_ops.py" in body
    assert "Verified zero denominator" in body
    assert "Small utilities" in body
    assert "Cache invalidation" in body


def test_cli_prints_plan_cards_after_creation_and_status_updates(tmp_path, monkeypatch, capsys):
    replies = iter([
        ("create_plan", {"title": "Math utils", "steps": ["Implement", "Test"]}),
        ("update_plan_step", {"step_id": 1, "status": "in_progress"}),
        ("update_plan_step", {"step_id": 1, "status": "completed", "notes": "Verified"}),
    ])

    def fake_llm(messages, override_config=None, tools=None):
        item = next(replies, None)
        if item is None:
            return {"status": "success", "content": "Ready for next step"}
        name, arguments = item
        return {"status": "success", "content": "", "tool_calls": [{
            "id": name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}

    monkeypatch.setattr(cli, "call_llm", fake_llm)
    cli.run_cli("work through math", sandbox_root=tmp_path, session_path=tmp_path / "session.json")
    output = capsys.readouterr().out
    assert output.count("Active plan") == 3
    assert "Math utils" in output
    assert "0/2 completed (0.0%)" in output
    assert "1/2 completed (50.0%)" in output
    assert "in_progress" in output
    assert "Verified" in output


async def test_tui_restores_existing_plan_progress_at_startup(tmp_path):
    create_plan(tmp_path, "Math", ["Implement", "Test"])
    execute_tool_result("update_plan_step", {"step_id": 1, "status": "completed"}, tmp_path)
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        assert app.query_one("#active_plan", VerticalScroll).display
        body = app.query_one("#plan_content", Static).content.renderable.plain
        assert "Math" in body
        assert "1/2 completed (50.0%)" in body


async def test_live_agent_turn_updates_the_plan_panel_from_background_events(tmp_path, monkeypatch):
    replies = iter([
        ("create_plan", {"title": "Math", "steps": ["Implement"]}),
        ("update_plan_step", {"step_id": 1, "status": "completed", "notes": "Verified"}),
    ])

    def fake_llm(messages, override_config=None, tools=None):
        item = next(replies, None)
        if item is None:
            return {"status": "success", "content": "Done"}
        name, arguments = item
        return {"status": "success", "content": "", "tool_calls": [{
            "id": name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}

    monkeypatch.setattr(tui_main, "call_llm", fake_llm)
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        prompt = app.query_one("#message_input", tui_main.PromptInput)
        prompt.text = "work through math"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        body = app.query_one("#plan_content", Static).content.renderable.plain
        assert "1/1 completed (100.0%)" in body
        assert "Verified" in body
        transcript = "\n".join(line.text for line in app.query_one("#messages", RichLog).lines)
        assert transcript.count("Active plan") == 2


async def test_tui_shows_empty_plan_panel_only_while_planning(tmp_path):
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test() as pilot:
        view = app.query_one("#active_plan", VerticalScroll)
        assert not view.display
        await pilot.press("ctrl+b")
        assert view.display
        assert "No active plan" in str(app.query_one("#plan_content", Static).content)
        await pilot.press("ctrl+b")
        assert not view.display


@pytest.mark.parametrize("depth", [0, 1])
async def test_tui_plan_tools_refresh_sidebar_and_emit_progress_cards(tmp_path, depth):
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        for name, arguments in [
            ("create_plan", {"title": "Math", "steps": ["Implement", "Test"]}),
            ("update_plan_step", {"step_id": 1, "status": "completed"}),
        ]:
            output, code = execute_tool_result(name, arguments, tmp_path)
            app._on_turn_event(TurnEvent(
                phase="tool_done", step=1, index=1, name=name, output=output, exit_code=code,
                depth=depth, agent="planner" if depth else "",
            ))
        view = app.query_one("#active_plan", VerticalScroll)
        assert view.display
        body = app.query_one("#plan_content", Static).content.renderable.plain
        assert "1/2 completed (50.0%)" in body
        assert "completed" in body
        transcript = "\n".join(line.text for line in app.query_one("#messages", RichLog).lines)
        assert "Active plan" in transcript
        assert "Math" in transcript
        before = body
        output, code = execute_tool_result(
            "update_plan_step", {"step_id": 9, "status": "completed"}, tmp_path
        )
        app._on_turn_event(TurnEvent(
            phase="tool_done", step=2, name="update_plan_step", output=output, exit_code=code,
            depth=depth,
        ))
        assert app.query_one("#plan_content", Static).content.renderable.plain == before


async def test_tui_corrupt_plan_is_reported_without_false_progress(tmp_path):
    (tmp_path / "PLAN.json").write_text("{broken")
    app = tui_main.RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        assert app.query_one("#active_plan", VerticalScroll).display
        content = app.query_one("#plan_content", Static).content
        assert "Cannot load active plan" in str(content.renderable)
        assert "completed" not in str(content.renderable)


async def test_shell_mode_does_not_show_active_plan(tmp_path):
    create_plan(tmp_path, "Math", ["Implement"])
    app = tui_main.RatchetApp(
        log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path, mode="shell"
    )
    async with app.run_test():
        assert not app.query_one("#active_plan", VerticalScroll).display
