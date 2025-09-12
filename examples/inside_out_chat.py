"""Demonstrate a multi-turn Inside Out-style group chat."""

from flask_group_chat import build_manager, split_content


def main() -> None:
    """Run an interactive multi-turn conversation for testing."""
    user, manager = build_manager()
    cursor = 0
    while True:
        try:
            prompt = input("用户: ")
        except EOFError:
            break
        if not prompt or prompt.lower() == "exit":
            break
        start = cursor
        user.initiate_chat(manager, message=prompt, clear_history=False)
        raw_replies = manager.groupchat.messages[start:]
        cursor = start + len(raw_replies)
        for m in raw_replies:
            for seg in split_content(m["content"]):
                print(f"{m['name']}: {seg}")


if __name__ == "__main__":
    main()
