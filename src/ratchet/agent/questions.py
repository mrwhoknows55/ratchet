import json

from ratchet.agent.context import AgentContext


def ask_user_question(
    ctx: AgentContext, question: str, options: list[str] | None = None,
) -> dict[str, str | int]:
    def error(message: str) -> dict[str, str | int]:
        return {"stdout": "", "stderr": f"Error: {message}", "exit_code": 1}

    if not isinstance(question, str) or not question.strip():
        return error("question must be a non-empty string.")
    if options is None:
        options = []
    if not isinstance(options, list) or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        return error("options must be a list of non-empty strings.")
    if ctx.ask_user is None:
        return error(
            "No interactive question handler is available; report the unresolved question."
        )
    question = question.strip()
    answer = ctx.ask_user(question, [option.strip() for option in options])
    if answer is None:
        return error(
            "Question cancelled; report the unresolved question without assuming an answer."
        )
    if not isinstance(answer, str) or not answer.strip():
        return error("No non-empty answer received; do not assume an answer.")
    return {
        "stdout": json.dumps({"question": question, "answer": answer.strip()}),
        "stderr": "",
        "exit_code": 0,
    }
