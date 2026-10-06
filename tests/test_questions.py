import json

import pytest

from ratchet.agent.context import AgentContext
from ratchet.agent.loop import run_turn
from ratchet.agent.tools import execute_tool_result


def test_question_returns_actual_answer_and_normalizes_input(tmp_path):
    seen = []

    def answer(question, options):
        seen.append((question, options))
        return " PostgreSQL "

    ctx = AgentContext(sandbox_root=tmp_path, ask_user=answer)
    output, code = execute_tool_result("ask_user_question", {
        "question": " Which storage? ", "options": [" SQLite ", "PostgreSQL"],
    }, ctx)
    assert code == 0
    assert seen == [("Which storage?", ["SQLite", "PostgreSQL"])]
    assert json.loads(output) == {"question": "Which storage?", "answer": "PostgreSQL"}


@pytest.mark.parametrize("arguments", [
    {"question": " "},
    {"question": "Storage?", "options": "SQLite"},
    {"question": "Storage?", "options": [""]},
])
def test_invalid_question_does_not_prompt_user(tmp_path, arguments):
    def unexpected_answer(*args):
        pytest.fail("Invalid question must not reach the frontend")

    ctx = AgentContext(sandbox_root=tmp_path, ask_user=unexpected_answer)
    output, code = execute_tool_result("ask_user_question", arguments, ctx)
    assert code == 1
    assert "Error" in output


@pytest.mark.parametrize("answer", [None, " "])
def test_cancelled_or_empty_answer_is_not_reported_as_success(tmp_path, answer):
    ctx = AgentContext(sandbox_root=tmp_path, ask_user=lambda *args: answer)
    output, code = execute_tool_result("ask_user_question", {"question": "Storage?"}, ctx)
    assert code == 1
    assert "answer" in output.lower() or "cancelled" in output.lower()


def test_question_without_interactive_handler_reports_error(tmp_path):
    output, code = execute_tool_result("ask_user_question", {"question": "Storage?"}, tmp_path)
    assert code == 1
    assert "interactive" in output.lower()


def test_planner_question_answer_reaches_its_isolated_history(tmp_path):
    parent_done = False
    child_answer = []
    seen_questions = []

    def tool_call(name, arguments):
        return {"status": "success", "content": "", "tool_calls": [{
            "id": name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}

    def fake_llm(messages, override_config=None, tools=None):
        nonlocal parent_done
        names = {s["function"]["name"] for s in tools}
        if "spawn_subagent" in names:
            if parent_done:
                return {"status": "success", "content": "Plan aligned"}
            parent_done = True
            return tool_call("spawn_subagent", {"task": "Design storage", "role": "planner"})
        assert "ask_user_question" in names
        if messages[-1]["role"] == "tool":
            child_answer.append(json.loads(messages[-1]["content"]))
            return {"status": "success", "content": "Use PostgreSQL"}
        return tool_call("ask_user_question", {"question": "Storage?", "options": ["SQLite"]})

    def answer(question, options):
        seen_questions.append((question, options))
        return "PostgreSQL"

    result = run_turn(fake_llm, "design", tmp_path, plan_mode=True, ask_user=answer)
    assert result.status == "ok"
    assert seen_questions == [("Storage?", ["SQLite"])]
    assert child_answer == [{"question": "Storage?", "answer": "PostgreSQL"}]


def test_parallel_researchers_are_not_offered_interactive_questions():
    from ratchet.agent.subagent import schema_for_role

    names = {s["function"]["name"] for s in schema_for_role("researcher")}
    assert "ask_user_question" not in names
