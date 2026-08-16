from ratchet import cli


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
