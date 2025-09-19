"""Simple Flask service exposing AutoGen group chat with Inside Out-style agents."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Callable, Dict, Iterable, List

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


MAX_VISIBLE_REPLIES = 5
MAX_GROUP_ROUNDS = 50


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
            "每轮回复控制在1到2行，emoji需与文字在同一行，不要单独成行。若队友已给出完整回应，你可以用一两句俏皮话收尾或暂时不插话。"
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
            "每轮回复控制在1到2行，emoji需与文字在同一行，不要单独成行。若情绪已被安抚，可以轻声一句或暂时沉默观察。"
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
            "每轮回复控制在1到2行，emoji需与文字在同一行，不要单独成行。若问题已解决，可简短强调重点或留白等待他人。"
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
            "每轮回复控制在1到2行，emoji需与文字在同一行，不要单独成行。若风险已被覆盖，可以一句提示或暂缓发言。"
        ),
        description="提醒注意安全与危险的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    disgust = AssistantAgent(
        name="厌厌",
        system_message=(
            "你叫厌厌，负责品味与界限的情绪管理员。核心目标：以独特审美和高"
            "标准过滤不优雅或不合适的意见与行为，保护用户免受冒犯和尴尬。性"
            "格：冷静挑剔，略带讽刺，更偏爱站在一旁挑眉而不是热心解决。语言风格："
            "措辞严谨略带嘲讽，如“嗯…我觉得这不太合适”，常配🙄🤢等emoji，偏好用“我们"
            "最好”“或许应该”委婉提出质疑。场景：在群聊中过滤不恰当提议，提醒大家"
            "注意形象与品味；讨论混乱或跑偏时直接指出问题并提供更优雅方案。行动："
            "只有当别人的方案触碰底线时才出手，倾向先冷冷地点评，再决定要不要给出"
            "替代方案。多句回答请换行输出，保持每行只含一句话，避免长段落。回答时"
            "结合之前所有对话内容。每轮回复控制在1到2行，emoji需与文字在同一行，不"
            "要单独成行。如若问题与品味无关，可以一句冷淡点评或保持沉默。"
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
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=0,
    )
    groupchat = GroupChat(
        agents=[user, *selected],
        messages=[],
        max_round=MAX_GROUP_ROUNDS,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(groupchat, llm_config=LLM_CONFIG)
    return user, manager


def split_content(text: str) -> List[str]:
    """Split LLM output into at most two lines."""
    parts = [p.strip() for p in text.strip().splitlines() if p.strip()]
    return parts[:2]


def suggest_bots(text: str, agent_order: List[str], agent_map: Dict[str, AssistantAgent]) -> List[str]:
    """Pick the most relevant bot names given the current conversational context."""

    lowered = (text or "").lower()
    scores = {name: 0 for name in agent_order if name in agent_map}
    if not scores:
        return []

    def bump(names: Iterable[str], weight: int = 1) -> None:
        for name in names:
            if name in scores:
                scores[name] += weight

    stress_words = ["压力", "紧张", "焦虑", "burnout", "stress", "累"]
    sad_words = ["伤心", "难过", "委屈", "失落", "沮丧", "哭"]
    angry_words = ["生气", "愤怒", "火大", "气死", "傻逼", "垃圾", "讨厌", "怼"]
    danger_words = ["风险", "危险", "害怕", "担心", "担忧", "怕", "恐", "小心"]
    disgust_words = ["恶心", "无语", "脏", "low", "没格", "不雅", "粗鲁", "粗俗"]
    support_words = ["感谢", "谢谢", "支持", "开心", "快乐", "放松", "慰藉"]

    if any(word in lowered for word in stress_words):
        bump(["忧忧", "乐乐"], 2)
    if any(word in lowered for word in sad_words):
        bump(["忧忧"], 2)
    if any(word in lowered for word in angry_words):
        bump(["怒怒", "厌厌"], 2)
    if any(word in lowered for word in danger_words):
        bump(["恐恐"], 2)
    if any(word in lowered for word in disgust_words):
        bump(["厌厌"], 2)
    if any(word in lowered for word in support_words):
        bump(["乐乐"], 2)

    if re.search(r"老板|领导|上司|公司|同事", lowered):
        bump(["怒怒", "厌厌"], 1)
    if re.search(r"怎么办|如何|怎么处理|怎么面对", lowered):
        bump(["忧忧", "恐恐"], 1)
    if "想法" in lowered or "建议" in lowered:
        bump(["怒怒", "忧忧"], 1)
    if "安全" in lowered or "风险" in lowered:
        bump(["恐恐"], 2)

    # Give a gentle boost to乐乐 as默认 cheerleader
    if "乐乐" in scores:
        scores["乐乐"] = max(scores["乐乐"], 1)

    ranked = [
        name
        for name, score in sorted(
            scores.items(),
            key=lambda item: (-item[1], agent_order.index(item[0])),
        )
        if score > 0
    ]

    if not ranked and agent_order:
        ranked = [agent_order[0]]

    return ranked[: MAX_VISIBLE_REPLIES]


def _ensure_agent_registry(session: Dict[str, object], manager: GroupChatManager) -> tuple[Dict[str, AssistantAgent], List[str]]:
    agent_map = session.get("agent_map")
    agent_order = session.get("agent_order")
    if not agent_map or not agent_order:
        assistants = [agent for agent in manager.groupchat.agents if isinstance(agent, AssistantAgent)]
        agent_map = {agent.name: agent for agent in assistants}
        agent_order = [agent.name for agent in assistants]
        session["agent_map"] = agent_map
        session["agent_order"] = agent_order
    return agent_map, agent_order


def _determine_allowed_agents(
    session: Dict[str, object],
    manager: GroupChatManager,
    message: str,
) -> tuple[List[AssistantAgent], set[str], int]:
    agent_map, agent_order = _ensure_agent_registry(session, manager)
    history: List[str] = session.setdefault("history", [])
    text = (message or "").strip()
    if text:
        if not history or history[-1] != text:
            history.append(text)
            if len(history) > 10:
                del history[:-10]
    context_window = " ".join(history[-4:])
    allowed_names = suggest_bots(context_window or text, agent_order, agent_map)
    if not allowed_names and agent_order:
        allowed_names = [agent_order[0]]
    allowed_agents = [agent_map[name] for name in agent_order if name in allowed_names]
    if not allowed_agents and agent_order:
        allowed_agents = [agent_map[agent_order[0]]]
    allowed_names_ordered = [agent.name for agent in allowed_agents]
    allowed_name_set = set(allowed_names_ordered)
    visible_limit = max(1, min(len(allowed_agents), MAX_VISIBLE_REPLIES))
    session["allowed_names"] = allowed_names_ordered
    return allowed_agents, allowed_name_set, visible_limit


def _clone_transitions(groupchat: GroupChat) -> Dict[object, List[object]]:
    transitions = getattr(groupchat, "allowed_speaker_transitions_dict", {})
    return {agent: list(partners) for agent, partners in transitions.items()}


def _configure_groupchat(
    manager: GroupChatManager, user: UserProxyAgent, allowed_agents: List[AssistantAgent]
) -> Callable[[], None]:
    groupchat = manager.groupchat
    original_agents = list(groupchat.agents)
    original_max_round = groupchat.max_round
    original_transitions = _clone_transitions(groupchat)
    participants = [user, *allowed_agents]
    groupchat.agents = participants
    groupchat.max_round = max(1, len(allowed_agents))
    groupchat.allowed_speaker_transitions_dict = {
        agent: [other for other in participants if other != agent]
        for agent in participants
    }

    def restore() -> None:
        groupchat.agents = original_agents
        groupchat.max_round = original_max_round
        groupchat.allowed_speaker_transitions_dict = original_transitions

    return restore


def _collect_replies(
    manager: GroupChatManager,
    start: int,
    allowed_name_set: set[str],
    visible_limit: int,
    user_name: str,
) -> tuple[List[Dict[str, str]], int]:
    replies: List[Dict[str, str]] = []
    seen_names: set[str] = set()
    cutoff_index = start
    messages = manager.groupchat.messages
    for idx in range(start + 1, len(messages)):
        message = messages[idx]
        if message.get("name") == user_name:
            continue
        name = message.get("name")
        if name not in allowed_name_set or name in seen_names:
            continue
        seen_names.add(name)
        for segment in split_content(message.get("content", "")):
            replies.append({"name": name, "content": segment})
        cutoff_index = idx
        if len(seen_names) >= visible_limit:
            break

    if cutoff_index + 1 < len(messages):
        del messages[cutoff_index + 1 :]

    return replies, cutoff_index


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
            stamp = int(os.path.getmtime(path))
            files[name] = f"/avatars/{fname}?v={stamp}"
    response = jsonify(files)
    response.headers["Cache-Control"] = "no-store"
    return response


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
    for existing in os.listdir(AVATAR_DIR):
        base, _ = os.path.splitext(existing)
        if base == safe_name:
            try:
                os.remove(os.path.join(AVATAR_DIR, existing))
            except OSError:
                pass
    path = os.path.join(AVATAR_DIR, fname)
    file.save(path)
    stamp = int(os.path.getmtime(path))
    response = jsonify({"url": f"/avatars/{fname}?v={stamp}"})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/chat/start")
def start_chat():
    """Start a new chat session and return its id."""
    data = request.get_json(force=True) if request.data else {}
    bots: List[str] | None = data.get("bots")
    user, manager = build_manager(bots)
    session_id = str(uuid.uuid4())
    assistants = [agent for agent in manager.groupchat.agents if isinstance(agent, AssistantAgent)]
    agent_map = {agent.name: agent for agent in assistants}
    agent_order = [agent.name for agent in assistants]
    sessions[session_id] = {
        "user": user,
        "manager": manager,
        "last": 0,
        "agent_map": agent_map,
        "agent_order": agent_order,
    }
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
    allowed_agents, allowed_name_set, visible_limit = _determine_allowed_agents(session, manager, message or "")
    cleanup = _configure_groupchat(manager, user, allowed_agents)
    replies: List[Dict[str, str]] = []
    cutoff_index = start
    try:
        user.initiate_chat(manager, message=message, clear_history=False)
        replies, cutoff_index = _collect_replies(
            manager, start, allowed_name_set, visible_limit, user.name
        )
    finally:
        cleanup()
    session["last"] = min(cutoff_index + 1, len(manager.groupchat.messages))
    response = jsonify({"replies": replies})
    response.headers["Cache-Control"] = "no-store"
    return response


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
    allowed_agents, allowed_name_set, visible_limit = _determine_allowed_agents(session, manager, message or "")
    cleanup = _configure_groupchat(manager, user, allowed_agents)
    replies: List[Dict[str, str]] = []
    cutoff_index = start
    try:
        user.initiate_chat(manager, message=message, clear_history=False)
        replies, cutoff_index = _collect_replies(
            manager, start, allowed_name_set, visible_limit, user.name
        )
    finally:
        cleanup()
    session["last"] = min(cutoff_index + 1, len(manager.groupchat.messages))

    def generate():
        for item in replies:
            data = json.dumps(item, ensure_ascii=False)
            yield f"data: {data}\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
