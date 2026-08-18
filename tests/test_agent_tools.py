from ratchet.agent import tools as agent_tools


def test_tool_schemas_include_list_files():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "list_files" in names


def test_execute_tool_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("")
    result = agent_tools.execute_tool("list_files", {}, tmp_path)
    assert result == "a.txt"


def test_execute_tool_list_files_empty_directory(tmp_path):
    result = agent_tools.execute_tool("list_files", {}, tmp_path)
    assert result == "(no output)"


def test_execute_tool_unknown_name(tmp_path):
    result = agent_tools.execute_tool("not_a_real_tool", {}, tmp_path)
    assert "unknown tool" in result.lower()
    assert "not_a_real_tool" in result


def test_run_agent_turn_returns_content_without_tool_call(tmp_path):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {"content": "hello", "model": "test-model", "status": "success"}

    reply = agent_tools.run_agent_turn(fake_call_llm, "hi", tmp_path)

    assert reply == "hello"


def test_run_agent_turn_short_circuits_on_non_success(tmp_path):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {"content": "[LM Studio Offline] ...", "model": "test-model", "status": "offline"}

    reply = agent_tools.run_agent_turn(fake_call_llm, "hi", tmp_path)

    assert reply == "[LM Studio Offline] ..."


def test_run_agent_turn_executes_tool_call_and_returns_final_reply(tmp_path):
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

    reply = agent_tools.run_agent_turn(fake_call_llm, "what files exist?", tmp_path)

    assert reply == "there is a.txt"
    assert len(calls) == 2
    tool_message = calls[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "a.txt"


def test_run_agent_turn_stops_after_max_iterations(tmp_path):
    def fake_call_llm(messages, override_config=None, tools=None):
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

    reply = agent_tools.run_agent_turn(fake_call_llm, "hi", tmp_path)

    assert "exceeded" in reply.lower()


def test_run_agent_turn_passes_override_config_through(tmp_path):
    seen = []

    def fake_call_llm(messages, override_config=None, tools=None):
        seen.append(override_config)
        return {"content": "ok", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(
        fake_call_llm, "hi", tmp_path, override_config={"model": {"name": "x"}}
    )

    assert seen == [{"model": {"name": "x"}}]
