import json
from uuid import UUID

import pytest
from textual.containers import VerticalScroll

from ratchet import cli
from ratchet.agent.context import AgentContext
from ratchet.agent.plan import create_plan, get_plan
from ratchet.agent.sessions import Session
from ratchet.agent.tools import SYSTEM_PROMPT, execute_tool_result
from ratchet.tui.main import RatchetApp, StatusBar
from ratchet.tui.tokens import TokenMeter


def test_new_sessions_isolate_plans_and_resume_saved_history(tmp_path):
    pointer = tmp_path / "session.json"
    first = Session.open(pointer, tmp_path)
    assert str(UUID(first.id)) == first.id
    first.messages.append({"role": "user", "content": "earlier"})
    first.usage = {"session_tokens": 500}
    first.save()
    create_plan(first.root, "First", ["Implement"])
    resumed = Session.open(pointer, tmp_path)
    assert resumed.id == first.id
    assert resumed.messages == first.messages
    assert resumed.usage == first.usage
    assert get_plan(resumed.root)["exit_code"] == 0
    pointer.unlink()
    second = Session.open(pointer, tmp_path)
    assert second.id != first.id
    assert second.messages == [{"role": "system", "content": SYSTEM_PROMPT}]
    assert second.usage == {}
    assert get_plan(second.root)["exit_code"] == 1
    assert get_plan(first.root)["exit_code"] == 0


def test_orphan_workspace_plan_does_not_attach_to_a_new_session(tmp_path):
    create_plan(tmp_path, "Unrelated", ["Old work"])
    session = Session.open(tmp_path / "session.json", tmp_path)
    assert get_plan(session.root)["exit_code"] == 1


def test_legacy_history_and_plan_migrate_together_once(tmp_path):
    pointer = tmp_path / "session.json"
    saved = [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "earlier"}]
    pointer.write_text(json.dumps(saved))
    create_plan(tmp_path, "Existing", ["Resume"])
    session = Session.open(pointer, tmp_path)
    assert session.messages == saved
    assert get_plan(session.root)["exit_code"] == 0
    pointer.unlink()
    fresh = Session.open(pointer, tmp_path)
    assert get_plan(fresh.root)["exit_code"] == 1


def test_delegated_planner_saves_to_its_parent_session(tmp_path):
    session = Session.open(tmp_path / "session.json", tmp_path)
    replies = iter([
        {"status": "success", "content": "", "tool_calls": [{
            "id": "plan", "type": "function", "function": {"name": "create_plan",
            "arguments": json.dumps({"title": "Delegated", "steps": ["Implement"]})},
        }]},
        {"status": "success", "content": "Planned"},
    ])
    ctx = AgentContext(sandbox_root=tmp_path, plan_root=session.root,
                       call_llm_fn=lambda *args, **kwargs: next(replies))
    output, code = execute_tool_result("spawn_subagent", {
        "task": "Plan implementation", "role": "planner",
    }, ctx)
    assert code == 0, output
    assert get_plan(session.root)["exit_code"] == 0
    assert not (tmp_path / "PLAN.json").exists()


async def test_reset_rotates_session_and_immediately_hides_old_plan(tmp_path):
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        old = app.session
        create_plan(old.root, "Old", ["Implement"])
        app._refresh_active_plan()
        assert app.query_one("#active_plan", VerticalScroll).display
        app.action_reset_memory()
        assert app.session.id != old.id
        assert not app.query_one("#active_plan", VerticalScroll).display
        assert get_plan(app.session.root)["exit_code"] == 1
        assert get_plan(old.root)["exit_code"] == 0
        restored = Session.open(app.session_path, tmp_path)
        assert restored.id == app.session.id


async def test_session_reset_is_ignored_during_an_active_turn(tmp_path):
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        old_id = app.session.id
        app.query_one("#status", StatusBar).start()
        app.action_reset_memory()
        assert app.session.id == old_id


@pytest.mark.parametrize("flag", ["--new", "--clear", "--reset"])
def test_cli_reset_cannot_resume_the_previous_plan(tmp_path, monkeypatch, flag):
    pointer = tmp_path / "session.json"
    previous = Session.open(pointer, tmp_path)
    create_plan(previous.root, "Previous", ["Old work"])
    monkeypatch.setattr(cli, "DEFAULT_SESSION_PATH", pointer)

    def fake_llm(messages, override_config=None, tools=None):
        assert not any("An active structured plan exists" in m["content"] for m in messages)
        return {"status": "success", "content": "Fresh", "usage": {
            "prompt_tokens": 100, "completion_tokens": 20,
        }, "context_window": 4096}

    monkeypatch.setattr(cli, "call_llm", fake_llm)
    cli.main([flag, "hello"])
    fresh = Session.open(pointer, tmp_path)
    assert fresh.id != previous.id
    assert get_plan(fresh.root)["exit_code"] == 1
    assert get_plan(previous.root)["exit_code"] == 0
    assert fresh.usage["session_tokens"] == 120


async def test_tui_resumes_saved_token_usage(tmp_path):
    session = Session.open(tmp_path / "session.json", tmp_path)
    session.usage = {"session_tokens": 500, "context_tokens": 120, "context_window": 1000}
    session.save()
    app = RatchetApp(log_path=tmp_path / "ratchet.log", sandbox_root=tmp_path)
    async with app.run_test():
        meter = app.query_one("#tokens", TokenMeter)
        assert "120 / 1,000" in meter.render_tokens().plain
        assert "session 500 tok" in meter.render_tokens().plain
