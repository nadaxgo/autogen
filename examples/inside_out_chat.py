"""Demonstrate a multi-turn Inside Out-style group chat."""

from flask_group_chat import build_manager


def main() -> None:
    """Run a simple multi-turn conversation for testing."""
    user, manager = build_manager()
    prompts = [
        "大家好，最近工作压力很大，总是提不起精神。",
        "我担心明天的演讲会出问题，你们有什么建议？",
    ]
    for p in prompts:
        result = user.initiate_chat(manager, message=p, clear_history=False)
        print(f"用户: {p}")
        print(f"群聊: {result.summary}\n")


if __name__ == "__main__":
    main()
