from ratchet.agent import config as agent_config
from ratchet.agent import tools as agent_tools


def test_tool_schemas_include_list_files():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "list_files" in names


def test_tool_schemas_include_new_tools():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "search_files" in names
    assert "read_files" in names
    assert "write_files" in names
    assert "delete_file" in names


def test_execute_tool_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("")
    result = agent_tools.execute_tool("list_files", {}, tmp_path)
    assert result == "a.txt"


def test_execute_tool_list_files_empty_directory(tmp_path):
    result = agent_tools.execute_tool("list_files", {}, tmp_path)
    assert result == "(no output)"


def test_execute_tool_read_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    result = agent_tools.execute_tool("read_files", {"path": "a.txt"}, tmp_path)
    assert result == "1| hello world"


def test_execute_tool_read_files_missing(tmp_path):
    result = agent_tools.execute_tool("read_files", {"path": "missing.txt"}, tmp_path)
    assert "not found" in result.lower()


def test_execute_tool_write_files(tmp_path):
    result = agent_tools.execute_tool(
        "write_files", {"path": "a.txt", "content": "hello"}, tmp_path
    )
    assert "wrote" in result.lower()
    assert (tmp_path / "a.txt").read_text() == "hello"


def test_execute_tool_delete_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = agent_tools.execute_tool("delete_file", {"path": "a.txt"}, tmp_path)
    assert "deleted" in result.lower()
    assert not (tmp_path / "a.txt").exists()


def test_execute_tool_search_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\n")
    result = agent_tools.execute_tool("search_files", {"pattern": "hello"}, tmp_path)
    assert "a.txt" in result


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


def test_run_agent_turn_calls_on_tool_call_before_and_after_execution(tmp_path):
    (tmp_path / "a.txt").write_text("")
    llm_calls = []
    notifications = []

    def fake_call_llm(messages, override_config=None, tools=None):
        llm_calls.append(messages)
        if len(llm_calls) == 1:
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

    agent_tools.run_agent_turn(
        fake_call_llm, "what files exist?", tmp_path, on_tool_call=notifications.append
    )

    assert notifications == [
        "tool: list_files running...",
        "tool: list_files -> a.txt",
    ]


def test_run_agent_turn_without_on_tool_call_does_not_error(tmp_path):
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


def _tool_call_forever(calls):
    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
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

    return fake_call_llm


def _use_config(monkeypatch, tmp_path, body):
    config_file = tmp_path / "config.toml"
    config_file.write_text(body)
    monkeypatch.setattr(agent_config, "CONFIG_FILE", config_file)


def test_run_agent_turn_reads_max_steps_from_config(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path, '[model]\nname = "test-model"\n\n[agent]\nmax_steps = 3\n')
    calls = []

    reply = agent_tools.run_agent_turn(_tool_call_forever(calls), "hi", tmp_path)

    assert len(calls) == 3
    assert "exceeded" in reply.lower()


def test_run_agent_turn_falls_back_to_default_max_steps_without_agent_section(
    monkeypatch, tmp_path
):
    _use_config(monkeypatch, tmp_path, '[model]\nname = "test-model"\n')
    calls = []

    agent_tools.run_agent_turn(_tool_call_forever(calls), "hi", tmp_path)

    assert len(calls) == agent_tools.DEFAULT_MAX_STEPS


def test_default_max_steps_is_twelve():
    assert agent_tools.DEFAULT_MAX_STEPS == 12
    assert agent_config.DEFAULT_CONFIG["agent"]["max_steps"] == 12


def test_tool_schemas_include_replace_in_file():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "replace_in_file" in names


def test_replace_in_file_schema_requires_all_three_arguments():
    schema = next(
        t for t in agent_tools.TOOL_SCHEMAS if t["function"]["name"] == "replace_in_file"
    )
    assert schema["function"]["parameters"]["required"] == ["path", "old_str", "new_str"]


def test_execute_tool_replace_in_file(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    result = agent_tools.execute_tool(
        "replace_in_file",
        {"path": "a.txt", "old_str": "beta", "new_str": "delta"},
        tmp_path,
    )
    assert (tmp_path / "a.txt").read_text() == "alpha\ndelta\n"
    assert "replace" in result.lower()


def test_tool_schemas_include_read_file_range():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "read_file_range" in names


def test_read_file_range_schema_requires_only_path():
    schema = next(
        t for t in agent_tools.TOOL_SCHEMAS if t["function"]["name"] == "read_file_range"
    )
    assert schema["function"]["parameters"]["required"] == ["path"]


def test_execute_tool_read_file_range(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    result = agent_tools.execute_tool(
        "read_file_range", {"path": "a.txt", "start_line": 2, "end_line": 3}, tmp_path
    )
    assert "2| two" in result
    assert "3| three" in result
    assert "one" not in result


def test_execute_tool_read_file_range_defaults_to_start_of_file(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = agent_tools.execute_tool("read_file_range", {"path": "a.txt"}, tmp_path)
    assert "1| one" in result


def test_system_prompt_leads_the_conversation(tmp_path):
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(messages)
        return {"content": "ok", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(fake_call_llm, "do a thing", tmp_path)

    assert calls[0][0] == {"role": "system", "content": agent_tools.SYSTEM_PROMPT}
    assert calls[0][1] == {"role": "user", "content": "do a thing"}


def test_system_prompt_stays_small():
    assert len(agent_tools.SYSTEM_PROMPT) < 600


def test_tool_descriptions_stay_terse():
    too_long = {
        tool["function"]["name"]: len(tool["function"]["description"])
        for tool in agent_tools.TOOL_SCHEMAS
        if len(tool["function"]["description"]) > 160
    }
    assert too_long == {}


def test_tool_schemas_include_run_command():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "run_command" in names


def test_run_command_schema_requires_command():
    schema = next(t for t in agent_tools.TOOL_SCHEMAS if t["function"]["name"] == "run_command")
    assert schema["function"]["parameters"]["required"] == ["command"]


def test_execute_tool_run_command(tmp_path):
    (tmp_path / "a.txt").write_text("hello from sandbox\n")
    result = agent_tools.execute_tool("run_command", {"command": "cat a.txt"}, tmp_path)
    assert "hello from sandbox" in result


def test_tool_schemas_include_file_search():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "file_search" in names


def test_file_search_schema_requires_only_pattern():
    schema = next(t for t in agent_tools.TOOL_SCHEMAS if t["function"]["name"] == "file_search")
    assert schema["function"]["parameters"]["required"] == ["pattern"]


def test_execute_tool_file_search(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    result = agent_tools.execute_tool("file_search", {"pattern": "*.py"}, tmp_path)
    assert result == "sub/c.py"


def test_execute_tool_file_search_scopes_to_path(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    result = agent_tools.execute_tool(
        "file_search", {"pattern": "*.py", "path": "sub"}, tmp_path
    )
    assert result == "sub/c.py"


def test_execute_tool_run_command_reports_exit_code_when_silent(tmp_path):
    result = agent_tools.execute_tool("run_command", {"command": "touch made.txt"}, tmp_path)
    assert result == "(no output, exit 0)"
    assert (tmp_path / "made.txt").exists()


def test_execute_tool_run_command_appends_nonzero_exit_code(tmp_path):
    result = agent_tools.execute_tool("run_command", {"command": "cat missing.txt"}, tmp_path)
    assert "[exit 1]" in result
    assert "missing.txt" in result
