from ratchet import cli


def test_main_delegates_to_tui(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_tui", lambda: called.append(True))
    cli.main()
    assert called == [True]
