import json
from pathlib import Path

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
    assert result == "a.txt (0 bytes)"


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


def test_run_agent_turn_appends_reply_to_passed_in_messages(tmp_path):
    def fake_call_llm(messages, override_config=None, tools=None):
        return {"content": "hello", "model": "test-model", "status": "success"}

    history = [{"role": "system", "content": "sys"}]
    reply = agent_tools.run_agent_turn(fake_call_llm, "hi", tmp_path, messages=history)

    assert reply == "hello"
    assert history[-2] == {"role": "user", "content": "hi"}
    assert history[-1] == {"role": "assistant", "content": "hello"}


def test_run_agent_turn_remembers_earlier_turns(tmp_path):
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(list(messages))
        return {"content": "ack", "model": "test-model", "status": "success"}

    history = [{"role": "system", "content": "sys"}]
    agent_tools.run_agent_turn(fake_call_llm, "first", tmp_path, messages=history)
    agent_tools.run_agent_turn(fake_call_llm, "second", tmp_path, messages=history)

    second_turn_messages = calls[1]
    assert {"role": "user", "content": "first"} in second_turn_messages
    assert {"role": "assistant", "content": "ack"} in second_turn_messages
    assert second_turn_messages[-1] == {"role": "user", "content": "second"}


def test_save_session_writes_messages_as_json(tmp_path):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    path = tmp_path / "session.json"

    agent_tools.save_session(messages, path)

    assert json.loads(path.read_text()) == messages


def test_save_session_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "session.json"

    agent_tools.save_session([{"role": "system", "content": "sys"}], path)

    assert path.exists()


def test_load_session_round_trips_saved_messages(tmp_path):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    path = tmp_path / "session.json"
    agent_tools.save_session(messages, path)

    assert agent_tools.load_session(path) == messages


def test_load_session_returns_none_when_missing(tmp_path):
    assert agent_tools.load_session(tmp_path / "missing.json") is None


def test_load_session_returns_none_on_invalid_json(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("not json")

    assert agent_tools.load_session(path) is None


def test_clear_session_removes_file(tmp_path):
    path = tmp_path / "session.json"
    agent_tools.save_session([{"role": "system", "content": "sys"}], path)

    agent_tools.clear_session(path)

    assert not path.exists()


def test_clear_session_missing_file_does_not_error(tmp_path):
    agent_tools.clear_session(tmp_path / "missing.json")


def test_run_agent_turn_executes_tool_call_and_returns_final_reply(tmp_path):
    (tmp_path / "a.txt").write_text("")
    calls = []

    def fake_call_llm(messages, override_config=None, tools=None):
        calls.append(list(messages))
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
    assert tool_message["content"] == "a.txt (0 bytes)"


def test_run_agent_turn_notifies_around_each_tool_call(tmp_path):
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
        fake_call_llm, "what files exist?", tmp_path, on_event=notifications.append
    )

    assert [(e.phase, e.name) for e in notifications] == [
        ("thinking", ""),
        ("tool_start", "list_files"),
        ("tool_done", "list_files"),
        ("thinking", ""),
    ]


def test_run_agent_turn_without_on_event_does_not_error(tmp_path):
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
    assert len(agent_tools.SYSTEM_PROMPT) < 2000


def test_system_prompt_loaded_from_prompts_file():
    prompt_path = Path(__file__).parent.parent / "prompts" / "system.md"
    assert agent_tools.SYSTEM_PROMPT == prompt_path.read_text(encoding="utf-8").strip()


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


def test_system_prompt_directs_uncovered_work_to_the_shell():
    assert "escape hatch" in agent_tools.SYSTEM_PROMPT


def test_system_prompt_discourages_repeat_calls():
    assert "Asking twice" in agent_tools.SYSTEM_PROMPT


def test_tool_schemas_include_rollback_file():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "rollback_file" in names


def test_execute_tool_rollback_file(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    agent_tools.execute_tool("write_files", {"path": "a.txt", "content": "v2"}, tmp_path)
    result = agent_tools.execute_tool("rollback_file", {"path": "a.txt"}, tmp_path)
    assert "Restored" in result
    assert (tmp_path / "a.txt").read_text() == "v1"


def test_execute_tool_rollback_file_without_snapshot(tmp_path):
    (tmp_path / "a.txt").write_text("v1")
    result = agent_tools.execute_tool("rollback_file", {"path": "a.txt"}, tmp_path)
    assert "No snapshot" in result


def test_tool_schemas_include_append_file():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "append_file" in names


def test_execute_tool_append_file(tmp_path):
    (tmp_path / "a.txt").write_text("line1\n")
    result = agent_tools.execute_tool(
        "append_file", {"path": "a.txt", "content": "line2\n"}, tmp_path
    )
    assert "Appended" in result
    assert (tmp_path / "a.txt").read_text() == "line1\nline2\n"


def test_tool_schemas_include_get_file_info():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "get_file_info" in names


def test_execute_tool_get_file_info(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    result = agent_tools.execute_tool("get_file_info", {"path": "a.txt"}, tmp_path)
    assert "lines: 2" in result
    assert "size: 8" in result


def test_run_command_schema_exposes_timeout():
    schema = next(t for t in agent_tools.TOOL_SCHEMAS if t["function"]["name"] == "run_command")
    assert "timeout" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["command"]


def test_execute_tool_run_command_honours_timeout(tmp_path):
    result = agent_tools.execute_tool(
        "run_command", {"command": "sleep 2", "timeout": 1}, tmp_path
    )
    assert "timed out" in result.lower()


def test_system_prompt_demands_exact_output():
    assert "exactly" in agent_tools.SYSTEM_PROMPT


def test_system_prompt_routes_shell_operators_to_a_script():
    assert "script" in agent_tools.SYSTEM_PROMPT


def test_tool_schemas_include_the_remaining_tools():
    names = [tool["function"]["name"] for tool in agent_tools.TOOL_SCHEMAS]
    assert "copy_file" in names
    assert "move_file" in names
    assert "search_web" in names
    assert "fetch_url" in names


def test_execute_tool_copy_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = agent_tools.execute_tool(
        "copy_file", {"source": "a.txt", "destination": "b.txt"}, tmp_path
    )
    assert "copied" in result.lower()
    assert (tmp_path / "b.txt").read_text() == "hello"


def test_execute_tool_move_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = agent_tools.execute_tool(
        "move_file", {"source": "a.txt", "destination": "b.txt"}, tmp_path
    )
    assert "moved" in result.lower()
    assert not (tmp_path / "a.txt").exists()


def test_execute_tool_delete_file_passes_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    result = agent_tools.execute_tool(
        "delete_file", {"path": "sub", "recursive": True}, tmp_path
    )
    assert "deleted" in result.lower()
    assert not (tmp_path / "sub").exists()


def test_execute_tool_list_files_scopes_to_path(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("")
    result = agent_tools.execute_tool("list_files", {"path": "sub"}, tmp_path)
    assert "c.txt" in result
    assert "a.txt" not in result


def test_execute_tool_search_files_scopes_to_path(tmp_path):
    (tmp_path / "a.txt").write_text("needle")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("needle")
    result = agent_tools.execute_tool(
        "search_files", {"pattern": "needle", "path": "sub"}, tmp_path
    )
    assert "c.txt" in result
    assert "a.txt" not in result


def test_execute_tool_search_web_delegates_to_the_web_module(tmp_path, monkeypatch):
    calls = {}

    def fake_search_web(query, max_results=5):
        calls["query"] = query
        calls["max_results"] = max_results
        return {"stdout": "1. Result", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent_tools, "search_web", fake_search_web)
    result = agent_tools.execute_tool(
        "search_web", {"query": "python asyncio", "max_results": 3}, tmp_path
    )

    assert calls == {"query": "python asyncio", "max_results": 3}
    assert result == "1. Result"


def test_execute_tool_fetch_url_delegates_to_the_web_module(tmp_path, monkeypatch):
    calls = {}

    def fake_fetch_url(url):
        calls["url"] = url
        return {"stdout": "# Page", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent_tools, "fetch_url", fake_fetch_url)
    result = agent_tools.execute_tool("fetch_url", {"url": "https://a.test"}, tmp_path)

    assert calls == {"url": "https://a.test"}
    assert result == "# Page"


def test_execute_tool_result_returns_output_and_exit_code(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    output, exit_code = agent_tools.execute_tool_result("read_files", {"path": "a.txt"}, tmp_path)
    assert output == "1| hello"
    assert exit_code == 0


def test_execute_tool_result_reports_a_failing_exit_code(tmp_path):
    output, exit_code = agent_tools.execute_tool_result(
        "read_files", {"path": "missing.txt"}, tmp_path
    )
    assert exit_code == 1
    assert "not found" in output.lower()


def test_execute_tool_result_reports_unknown_tool(tmp_path):
    output, exit_code = agent_tools.execute_tool_result("nope", {}, tmp_path)
    assert exit_code == 1
    assert "unknown tool" in output.lower()


def _tool_turn_events(tmp_path):
    events = []
    calls = {"n": 0}

    def fake_call_llm(messages, override_config=None, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "done", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(
        fake_call_llm, "what files exist?", tmp_path, on_event=events.append
    )
    return events


def test_run_agent_turn_emits_thinking_start_and_done_phases(tmp_path):
    (tmp_path / "a.txt").write_text("")
    phases = [event.phase for event in _tool_turn_events(tmp_path)]
    assert phases == ["thinking", "tool_start", "tool_done", "thinking"]


def test_run_agent_turn_done_event_carries_name_output_and_exit_code(tmp_path):
    (tmp_path / "a.txt").write_text("")
    done = [e for e in _tool_turn_events(tmp_path) if e.phase == "tool_done"][0]
    assert done.name == "list_files"
    assert done.output == "a.txt (0 bytes)"
    assert done.exit_code == 0
    assert done.elapsed >= 0


def test_run_agent_turn_numbers_each_step(tmp_path):
    (tmp_path / "a.txt").write_text("")
    events = _tool_turn_events(tmp_path)
    assert [e.step for e in events] == [1, 1, 1, 2]


def test_run_agent_turn_reports_a_failing_tool_exit_code(tmp_path):
    events = []
    calls = {"n": 0}

    def fake_call_llm(messages, override_config=None, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"path": "missing.txt"}',
                        },
                    }
                ],
            }
        return {"content": "done", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(fake_call_llm, "read it", tmp_path, on_event=events.append)

    done = [e for e in events if e.phase == "tool_done"][0]
    assert done.exit_code == 1


def test_run_agent_turn_start_event_carries_the_arguments(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    events = []

    def fake_call_llm(messages, override_config=None, tools=None):
        if len(messages) < 3:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
            }
        return {"content": "done", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(fake_call_llm, "read it", tmp_path, on_event=events.append)

    start = [e for e in events if e.phase == "tool_start"][0]
    assert start.arguments == {"path": "a.txt"}


def test_run_agent_turn_numbers_tool_calls_across_rounds(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    events = []
    calls = {"n": 0}

    def fake_call_llm(messages, override_config=None, tools=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {
                        "id": f"call_{calls['n']}",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "done", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(fake_call_llm, "look twice", tmp_path, on_event=events.append)

    indexes = [e.index for e in events if e.phase == "tool_done"]
    assert indexes == [1, 2]


def test_run_agent_turn_numbers_parallel_tool_calls_in_one_round(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    events = []
    calls = {"n": 0}

    def fake_call_llm(messages, override_config=None, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "model": "test-model",
                "status": "success",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "list_files", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "list_files", "arguments": "{}"}},
                ],
            }
        return {"content": "done", "model": "test-model", "status": "success"}

    agent_tools.run_agent_turn(fake_call_llm, "look", tmp_path, on_event=events.append)

    assert [e.index for e in events if e.phase == "tool_start"] == [1, 2]
