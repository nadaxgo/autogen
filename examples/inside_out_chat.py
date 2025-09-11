"""Demonstrate a multi-turn Inside Out-style group chat."""

from flask_group_chat import build_manager


def main() -> None:
    """Run an interactive multi-turn conversation for testing."""
    user, manager = build_manager()
    first = input("用户: ")
    user.initiate_chat(manager, message=first, clear_history=False)


if __name__ == "__main__":
    main()
