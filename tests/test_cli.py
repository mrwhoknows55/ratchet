from ratchet import cli
from ratchet.agent.events import TurnEvent


def test_main_with_no_args_runs_chat_mode(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_tui", lambda mode: called.append(mode))
    cli.main([])
    assert called == ["chat"]


def test_main_with_shell_arg_runs_shell_mode(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_tui", lambda mode: called.append(mode))
    cli.main(["shell"])
    assert called == ["shell"]


def test_main_with_prompt_runs_cli_chat_mode(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_cli", lambda prompt, mode: called.append((prompt, mode)))
    cli.main(["list", "the files"])
    assert called == [("list the files", "chat")]


def test_main_with_shell_and_prompt_runs_cli_shell_mode(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_cli", lambda prompt, mode: called.append((prompt, mode)))
    cli.main(["shell", "ls -a"])
    assert called == [("ls -a", "shell")]


def test_run_cli_chat_prints_tool_chain_and_reply(monkeypatch, tmp_path, capsys):
    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event, messages):
        on_event(
            TurnEvent(phase="tool_start", step=1, name="read_file", arguments={"path": "a.txt"})
        )
        on_event(TurnEvent(phase="tool_end", step=1, name="read_file", output="hi", elapsed=0.2))
        return "done"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    cli.run_cli("read a.txt", sandbox_root=tmp_path, session_path=tmp_path / "session.json")
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "done" in out


def test_run_cli_chat_announces_a_tool_before_it_runs(monkeypatch, tmp_path, capsys):
    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event, messages):
        on_event(
            TurnEvent(phase="tool_start", step=1, name="read_file", arguments={"path": "a.txt"})
        )
        on_event(TurnEvent(phase="tool_done", step=1, name="read_file", output="hi", elapsed=0.2))
        return "done"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    cli.run_cli("read a.txt", sandbox_root=tmp_path, session_path=tmp_path / "session.json")
    out = capsys.readouterr().out
    assert "running read_file" in out


def test_run_cli_chat_persists_session_across_calls(monkeypatch, tmp_path):
    calls = []

    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event, messages):
        calls.append(list(messages))
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "ack"})
        return "ack"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    session_path = tmp_path / "session.json"

    cli.run_cli("first", sandbox_root=tmp_path, session_path=session_path)
    cli.run_cli("second", sandbox_root=tmp_path, session_path=session_path)

    second_call_messages = calls[1]
    assert {"role": "user", "content": "first"} in second_call_messages
    assert {"role": "assistant", "content": "ack"} in second_call_messages


def test_main_with_reset_flag_and_no_prompt_clears_session_and_exits(monkeypatch, capsys):
    cleared = []
    monkeypatch.setattr(cli, "clear_session", lambda path: cleared.append(path))
    ran = []
    monkeypatch.setattr(cli, "run_cli", lambda prompt, mode: ran.append(prompt))
    monkeypatch.setattr(cli, "run_tui", lambda mode: ran.append("tui"))

    cli.main(["--clear"])

    assert cleared == [cli.DEFAULT_SESSION_PATH]
    assert ran == []
    assert "cleared" in capsys.readouterr().out.lower()


def test_main_with_new_flag_and_prompt_clears_then_runs(monkeypatch):
    cleared = []
    monkeypatch.setattr(cli, "clear_session", lambda path: cleared.append(path))
    called = []
    monkeypatch.setattr(cli, "run_cli", lambda prompt, mode: called.append((prompt, mode)))

    cli.main(["--new", "What", "is", "my", "name?"])

    assert cleared == [cli.DEFAULT_SESSION_PATH]
    assert called == [("What is my name?", "chat")]


def test_main_reset_alias_flag_also_clears(monkeypatch, capsys):
    cleared = []
    monkeypatch.setattr(cli, "clear_session", lambda path: cleared.append(path))

    cli.main(["--reset"])

    assert cleared == [cli.DEFAULT_SESSION_PATH]


def test_run_cli_shell_prints_output_and_exit_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda command, root: {"stdout": "a.txt\n", "stderr": "", "exit_code": 0},
    )
    cli.run_cli("ls", mode="shell", sandbox_root=tmp_path)
    out = capsys.readouterr().out
    assert "a.txt" in out
    assert "exit 0" in out


def test_run_cli_chat_indents_nested_subagent_calls(monkeypatch, tmp_path, capsys):
    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event, messages):
        on_event(
            TurnEvent(
                phase="tool_start", step=1, index=1, name="read_files",
                arguments={"path": "a.txt"}, depth=1, agent="researcher",
            )
        )
        on_event(
            TurnEvent(
                phase="tool_done", step=1, index=1, name="read_files", output="hi",
                elapsed=0.2, depth=1, agent="researcher",
            )
        )
        return "done"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    cli.run_cli("read a.txt", sandbox_root=tmp_path, session_path=tmp_path / "session.json")
    lines = [line for line in capsys.readouterr().out.splitlines() if "read_files" in line]
    assert lines
    assert all(line.startswith("    ") for line in lines)
    assert any("running" in line for line in lines)


def test_run_cli_chat_frames_a_delegation_with_a_header_and_footer(monkeypatch, tmp_path, capsys):
    output = "notes.txt mentions retry\n[researcher · 4.1s · 3/8 steps · 360 tok · via read_files]"

    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event, messages):
        args = {"task": "which file mentions retry", "role": "researcher"}
        on_event(
            TurnEvent(phase="tool_start", step=1, index=1, name="spawn_subagent", arguments=args)
        )
        on_event(
            TurnEvent(
                phase="tool_done", step=1, index=1, name="spawn_subagent", arguments=args,
                output=output, exit_code=0, elapsed=4.1,
            )
        )
        return "done"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    cli.run_cli("find retry", sandbox_root=tmp_path, session_path=tmp_path / "session.json")
    out = capsys.readouterr().out
    assert "researcher" in out
    assert "which file mentions retry" in out
    assert "running" in out
    assert "done" in out
    assert "3/8 steps" in out
    assert "notes.txt mentions retry" in out
