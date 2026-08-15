def run() -> None:
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        print(f"user said {user_input}")
