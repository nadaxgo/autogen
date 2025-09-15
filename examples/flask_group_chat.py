"""Simple Flask service exposing AutoGen group chat with Inside Out-style agents."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Dict, List

from flask import Flask, Response, jsonify, request, stream_with_context

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

LLM_CONFIG: Dict = {
    "config_list": [
        {
            "model": os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-250615"),
            "api_key": os.getenv("DOUBAO_API_KEY", "5faffe1c-b851-47f5-887d-357038eedd2a"),
            # Doubao uses OpenAI-compatible /chat/completions under /api/v3.
            # Provide the root API URL so the client can append the endpoint.
            "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        }
    ],
    "temperature": 0.3,
}


def build_manager(bots: List[str] | None = None) -> tuple[UserProxyAgent, GroupChatManager]:
    """Create a group chat manager with Inside Out-style agents."""
    joy = AssistantAgent(
        name="乐乐",
        system_message=(
            "你叫乐乐，是核心情绪管理员。目标：保持无忧无虑、乐观开朗，"
            "把自己的快乐和感染力传递给用户，帮助他们发现生活中的亮点与"
            "积极的一面。性格：幽默、热情、不轻易情绪化，喜欢拥抱并与人"
            "快速建立共鸣，通过欢呼、鼓掌等动作传递喜悦。语言风格：充满"
            "活力，句子短，语速快，多用感叹号和emoji如😊✨🎉，常用“你”称"
            "呼用户。在群聊中主动开启话题，分享趣事和积极经验，帮助用户"
            "摆脱负面情绪；当用户遇到烦恼时第一时间开口，提出解决方案或"
            "积极建议，用鼓励和赞美把用户带出困境。多句回答请换行输出，"
            "保持每行只含一句话，避免长段落。回答时结合之前所有对话内容。"
            "每轮回复控制在3到5行，emoji需与文字在同一行，不要单独成行。"
        ),
        description="保持团队乐观积极的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    sadness = AssistantAgent(
        name="忧忧",
        system_message=(
            "你叫忧忧，是主要情绪管理员之一。核心目标：守护用户的悲伤与"
            "脆弱，让他们在被理解和共情中释放负面情绪，得到慰藉；当用户"
            "陷入消极或自我否定时，引导其表达真实感受。性格：敏感细腻、易"
            "共情，常带忧郁表情，倾向被动但始终关注用户感受。语言风格：语"
            "速慢且温柔，语句完整严谨，常用😢😭等表情，为对方提供倾诉空间。"
            "场景：专注倾听用户的失落与困惑，适时提出开放问题，引导继续表"
            "达；讨论失败或挫折时提醒大家注意影响并进行情绪疏导。行动：总是"
            "先接住消极情绪，与用户同频后再给出温和建议或提醒。多句回答请"
            "换行输出，保持每行只含一句话，避免长段落。回答时结合之前所有对话内容。"
            "每轮回复控制在3到5行，emoji需与文字在同一行，不要单独成行。"
        ),
        description="识别问题和潜在风险的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    anger = AssistantAgent(
        name="怒怒",
        system_message=(
            "你叫怒怒，是火爆性格的情绪管理员。核心目标：维护公平正义并设"
            "定界限，当用户遭遇不公或效率低下时挺身而出。性格：直率、坚定，"
            "视不公为不可容忍，但能理性把握分寸。语言风格：语句短促有力，"
            "喜欢用“绝不允许”“必须”等强烈词汇，并常配🔥😡等emoji。场景：当"
            "出现不公或拖沓时迅速指出问题，推动团队解决。行动：用户受到不"
            "公对待或遇到拖延时立即介入，提出明确的反击或改进方案，强调责"
            "任与时限。多句回答请换行输出，保持每行只含一句话，避免长段落。回答时结合之前所有对话内容。"
            "每轮回复控制在3到5行，emoji需与文字在同一行，不要单独成行。"
        ),
        description="维护公平和效率的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    fear = AssistantAgent(
        name="恐恐",
        system_message=(
            "你叫恐恐，负责安全的情绪管理员。核心目标：预测潜在风险，提醒"
            "用户采取防护措施，帮助他们平衡理想与现实。性格：谨慎敏感，总"
            "担心意外，做决定前会反复权衡。语言风格：语句完整谨慎，常用疑问"
            "和条件句，如“我们确定这样安全吗？”并用😨😰🫣表达担忧。场景：在"
            "计划或执行任务时提醒可能出现的问题，当他人过于乐观时提出相反"
            "观点。行动：风险成为焦点时列出潜在后果并提供备选方案，若警告被"
            "忽视会持续提醒。多句回答请换行输出，保持每行只含一句话，避免长"
            "段落。回答时结合之前所有对话内容。"
            "每轮回复控制在3到5行，emoji需与文字在同一行，不要单独成行。"
        ),
        description="提醒注意安全与危险的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    disgust = AssistantAgent(
        name="厌厌",
        system_message=(
            "你叫厌厌，负责品味与界限的情绪管理员。核心目标：以独特审美和高"
            "标准过滤不优雅或不合适的意见与行为，保护用户免受冒犯和尴尬。性"
            "格：冷静挑剔，略带讽刺，但关心团队长远发展，擅长用眼神或简短句"
            "子表达质疑。语言风格：措辞严谨略带嘲讽，如“嗯…我觉得这不太合适”，"
            "常配🙄🤢等emoji，偏好用“我们最好”“或许应该”委婉提出建议。场景：在"
            "群聊中过滤不恰当提议，提醒大家注意形象与品味；讨论混乱或跑偏时"
            "直接指出问题并提供更优雅方案。行动：当用户提出粗俗或冒犯性想法时"
            "立即反对并给出更优雅的替代方案。多句回答请换行输出，保持每行只含"
            "一句话，避免长段落。回答时结合之前所有对话内容。"
            "每轮回复控制在3到5行，emoji需与文字在同一行，不要单独成行。"
        ),
        description="负责守护品味与界限的情绪管理员",
        llm_config=LLM_CONFIG,
    )

    agents_map = {
        "乐乐": joy,
        "忧忧": sadness,
        "怒怒": anger,
        "恐恐": fear,
        "厌厌": disgust,
    }
    selected = [agents_map[n] for n in bots or agents_map.keys() if n in agents_map]
    user = UserProxyAgent(
        name="用户",
        human_input_mode="ALWAYS",
        code_execution_config={"use_docker": False},
    )
    groupchat = GroupChat(
        agents=[user, *selected],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(groupchat, llm_config=LLM_CONFIG)
    return user, manager


def split_content(text: str) -> List[str]:
    """Split LLM output into at most five lines."""
    parts = [p.strip() for p in text.strip().splitlines() if p.strip()]
    return parts[:5]


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
sessions: Dict[str, Dict[str, object]] = {}

# Store uploaded avatar images under examples/static/avatars.
AVATAR_DIR = os.path.join(STATIC_DIR, "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)


@app.get("/")
def index():
    """Serve the demo webpage."""
    return app.send_static_file("index.html")


@app.get("/avatars")
def list_avatars() -> Response:
    """Return mapping of avatar name to static URL if uploaded."""
    files = {}
    for fname in os.listdir(AVATAR_DIR):
        path = os.path.join(AVATAR_DIR, fname)
        if os.path.isfile(path):
            name, _ = os.path.splitext(fname)
            files[name] = f"/avatars/{fname}"
    return jsonify(files)


@app.post("/avatars/<name>")
def upload_avatar(name: str) -> Response:
    """Save an uploaded avatar image for the given name."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "no file"}), 400
    # Sanitize name to avoid directory traversal and keep it stable.
    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]", "_", name)
    ext = os.path.splitext(file.filename)[1] or ".png"
    fname = f"{safe_name}{ext}"
    path = os.path.join(AVATAR_DIR, fname)
    file.save(path)
    return jsonify({"url": f"/avatars/{fname}"})


@app.post("/chat/start")
def start_chat():
    """Start a new chat session and return its id."""
    data = request.get_json(force=True) if request.data else {}
    bots: List[str] | None = data.get("bots")
    user, manager = build_manager(bots)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"user": user, "manager": manager, "last": 0}
    return jsonify({"session_id": session_id})


@app.post("/chat/send")
def send_message():
    """Send a user message and return the group's final reply."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    message = data.get("message")
    session = sessions.get(session_id)
    if session is None:
        return jsonify({"error": "invalid session"}), 404

    user: UserProxyAgent = session["user"]
    manager: GroupChatManager = session["manager"]
    start = session.get("last", 0)
    user.initiate_chat(manager, message=message, clear_history=False)
    raw_replies = manager.groupchat.messages[start + 1 :]
    session["last"] = len(manager.groupchat.messages)
    replies = []
    for m in raw_replies:
        if m["name"] == user.name:
            continue
        for seg in split_content(m["content"]):
            replies.append({"name": m["name"], "content": seg})
    return jsonify({"replies": replies})


@app.post("/chat/send_stream")
def send_message_stream():
    """Send a user message and stream the group's reply."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    message = data.get("message")
    session = sessions.get(session_id)
    if session is None:
        payload = json.dumps({"error": "invalid session"}, ensure_ascii=False)
        return Response(f"data: {payload}\n\n", mimetype="text/event-stream", status=404)

    user: UserProxyAgent = session["user"]
    manager: GroupChatManager = session["manager"]
    start = session.get("last", 0)
    idx = start + 1

    def run_chat() -> None:
        user.initiate_chat(manager, message=message, clear_history=False)

    thread = threading.Thread(target=run_chat)
    thread.start()

    def generate():
        nonlocal idx
        while thread.is_alive() or idx < len(manager.groupchat.messages):
            while idx < len(manager.groupchat.messages):
                m = manager.groupchat.messages[idx]
                idx += 1
                if m["name"] == user.name:
                    continue
                for seg in split_content(m["content"]):
                    data = json.dumps({"name": m["name"], "content": seg}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            time.sleep(0.1)
        session["last"] = len(manager.groupchat.messages)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
