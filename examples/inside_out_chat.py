"""Demonstrate a multi-turn Inside Out-style group chat."""

from flask_group_chat import build_manager


def main() -> None:
    """Run an interactive multi-turn conversation for testing."""
    user, manager = build_manager()
    while True:
        try:
            prompt = input("用户: ")
        except EOFError:
            break
        if not prompt or prompt.lower() == "exit":
            break
        user.initiate_chat(manager, message=prompt, clear_history=False)


if __name__ == "__main__":
    main()
