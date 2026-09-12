from ratchet import cli
from ratchet.agent.tools import TurnEvent


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
    def fake_run_agent_turn(call_llm_fn, text, sandbox_root, override_config, on_event):
        on_event(
            TurnEvent(phase="tool_start", step=1, name="read_file", arguments={"path": "a.txt"})
        )
        on_event(TurnEvent(phase="tool_end", step=1, name="read_file", output="hi", elapsed=0.2))
        return "done"

    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)
    cli.run_cli("read a.txt", sandbox_root=tmp_path)
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "done" in out


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
