import argparse

from ratchet.tui.main import main as run_tui


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ratchet")
    parser.add_argument("mode", nargs="?", choices=["shell"], default=None)
    args = parser.parse_args(argv)
    run_tui(mode="shell" if args.mode == "shell" else "chat")


if __name__ == "__main__":
    main()
