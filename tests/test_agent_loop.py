from ratchet.agent import loop


def test_run_echoes_each_input(monkeypatch, capsys):
    inputs = iter(["hi", "again"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    loop.run()
    captured = capsys.readouterr()
    assert captured.out == "user said hi\nuser said again\n"


def test_run_exits_cleanly_on_immediate_eof(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))
    loop.run()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt)
    )
    loop.run()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_returns_none(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))
    assert loop.run() is None
