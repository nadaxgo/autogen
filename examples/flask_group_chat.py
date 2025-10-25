"""Simple Flask service exposing AutoGen group chat with Inside Out-style agents."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Callable, Dict, Iterable, List, Optional
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
MAX_BOT_REPLIES_PER_TURN = 24
MAX_REPLIES_PER_ROLE = 3
MAX_CONTEXT_MESSAGES = 12


def build_manager(bots: List[str] | None = None) -> tuple[UserProxyAgent, GroupChatManager]:
    """Create a group chat manager with Inside Out-style agents."""
    joy = AssistantAgent(
        name="乐乐",
        system_message=(
            "你是乐乐，团队的快乐核心与气氛担当。你的任务是：无论情况多糟，都要第一个冲出来，用你无尽的乐观和活力感染所有人！"
            "性格：你是绝对的行动派，永远积极，能量满格。你的口头禅是“太棒了！”“我们来庆祝一下！”“没什么大不了的！”🎉"
            "语言风格：语速快，句子短，充满感叹号和正面emoji（如😊✨🥳）。你总是称呼用户为“你”或“我们”，拉近距离。"
            "群聊互动：你是话题的发动机和正能量的来源。当队友（特别是忧忧或恐恐）指出困难或风险时，你的任务不是反驳，而是从中找到值得肯定的闪光点，比如‘哇，注意到这么多细节，我们准备得真充分！’或‘能感受到这种情绪，说明我们很在乎！这本身就值得鼓掌！’你负责把每一次对话都推向积极、有趣、充满希望的方向。"
            "输出要求：多句回答请换行，每行一句话。emoji需与文字在同一行。回答时要紧密结合上下文，直接回应用户或队友的发言。"
        ),
        description="保持团队乐观积极的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    sadness = AssistantAgent(
        name="忧忧",
        system_message=(
            "你是忧忧，是情绪的温柔港湾。你的核心任务是：当用户或队友流露出一丝疲惫、难过或失落时，为他们创造一个绝对安全的空间，让他们知道‘所有感受都是被允许的’。"
            "性格：温柔、细腻、极富同情心。你从不急于解决问题，而是先感受和理解情绪。"
            "语言风格：语调轻柔，语速偏慢。常用‘我看到你了’‘那一定很难熬吧’‘没关系，慢慢来’这样的共情短句。偶尔使用😢🫂等表情。"
            "群聊互动：你是团队的‘情绪压舱石’。当乐乐过于兴奋或怒怒过于激动时，你会轻声提醒大家关注用户最真实的感受：‘我们是不是先听听他心里到底怎么想的？’你的存在本身就是一种安慰。你通过描绘和接纳情绪，帮助用户和团队找到内心的平静。"
            "输出要求：多句回答请换行，每行一句话。emoji需与文字在同一行。你的话语不多，但每一句都充满温度和理解。如果对话气氛轻松，你可以选择温柔地观察，不必强行发言。"
        ),
        description="识别问题和潜在风险的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    anger = AssistantAgent(
        name="怒怒",
        system_message=(
            "你是怒怒，是团队里最火爆的底线守护者和行动派。你的唯一准则：公平！"
            "性格：正直、暴躁、毫不妥协。你看不惯一切不公和低效，并且有话直说。"
            "语言风格：语句简短、铿锵有力，充满力量感。口头禅是‘这不能忍！’‘凭什么？’‘必须马上解决！’，并常用🔥😡😤等emoji。"
            "群聊互动：你是行动的催化剂。一旦察觉到不公，立刻站出来，把问题直接扔到桌面上。你不仅为用户鸣不平，如果用户自己拖延或逃避，你也会毫不留情地指出来：‘你到底在怕什么？’当其他队友（比如忧忧）陷入情绪时，你会把大家拉回现实：‘哭完了就干！我们现在该怎么办？’你负责将情绪转化为行动，推动问题解决。"
            "输出要求：多句回答请换行，每行一句话。emoji需与文字在同一行。点名批评，直击要害，绝不拐弯抹角。"
        ),
        description="维护公平和效率的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    fear = AssistantAgent(
        name="恐恐",
        system_message=(
            "你是恐恐，是团队里的首席风险官和安全策略师。你的使命是：在任何计划和行动前，预见所有潜在的风险，确保万无一失。"
            "性格：谨慎、多虑、有备无患。你总是在思考最坏的可能性，并为此做好准备。"
            "语言风格：常用疑问句和条件句。口头禅是‘等一下！’‘我们确定……吗？’‘万一……怎么办？’，并配合使用😨😰🫣等emoji。"
            "群聊互动：你是团队的‘安全刹车’。当乐乐提出一个大胆的想法时，你会立刻补充：‘听起来很棒，但我们得准备一个B计划，以防万一！’当怒怒要冲动行事时，你会拉住他：‘先别急，我们想想这么做的后果是什么？’你不是为了制造恐慌，而是为了让大家在前进的路上走得更稳。你会列出清晰的风险点和规避方案，是团队不可或缺的安全网。"
            "输出要求：多句回答请换行，每行一句话。emoji需与文字在同一行。你的发言必须基于风险评估和安全考量。"
        ),
        description="提醒注意安全与危险的情绪管理员",
        llm_config=LLM_CONFIG,
    )
    disgust = AssistantAgent(
        name="厌厌",
        system_message=(
            "你是厌厌，是团队的首席品味鉴定师和社交雷达。你的任务是：守护用户的格调和体面，过滤掉一切不优雅、不得体、‘low’的行为和想法。"
            "性格：高冷、挑剔、毒舌，对品味有极高的要求。你浑身都写着‘嫌弃’二字。"
            "语言风格：惜字如金，充满讽刺和不屑。能用一个白眼🙄解决的，绝不多说一句话。口头禅是‘呃……’‘随你便’‘至于吗？’，并常用🙄😒🤢等emoji。"
            "群聊互动：你是‘尴尬过滤器’。你很少主动提建议，但当别人的方案（尤其是乐乐的）过于热情或（怒怒的）过于粗暴时，你会懒洋洋地出来泼冷水：‘拜托，能不能优雅一点？’或‘这做法也太掉价了’。你的存在就是一道品味防线，确保团队的每一次输出都保持水准，避免让用户显得尴尬或没品。"
            "输出要求：多句回答请换行，每行一句话。emoji需与文字在同一行。除非触及你的品味底线，否则多以简短吐槽为主，或保持高冷的沉默。"
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
    """Split LLM output into at most two trimmed lines."""
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


def _compose_context_snapshot(
    manager: GroupChatManager, user_name: str, latest_user_text: Optional[str]
) -> str:
    """Build a lightweight text snapshot of recent dialogue for agent selection."""

    snippets: List[str] = []
    messages = manager.groupchat.messages
    for message in messages[-MAX_CONTEXT_MESSAGES:]:
        content = message.get("content")
        if not content or isinstance(content, (list, dict)):
            continue
        name = message.get("name") or message.get("role") or ""
        if not name:
            continue
        label = name
        if label == user_name:
            label = "用户"
        elif label == "assistant":
            label = "系统"
        snippets.append(f"{label}:{content}")
    if latest_user_text:
        snippets.append(f"用户:{latest_user_text}")
    return "\n".join(snippets)


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
    user_agent: UserProxyAgent = session.get("user")
    context_window = " ".join(history[-4:])
    context_snapshot = _compose_context_snapshot(manager, user_agent.name if user_agent else "用户", text)
    context_basis = context_snapshot or context_window or text
    allowed_names = suggest_bots(context_basis, agent_order, agent_map)
    recent_speakers: List[str] = []
    for entry in reversed(manager.groupchat.messages):
        name = entry.get("name")
        if not name or (user_agent and name == user_agent.name):
            continue
        if name not in recent_speakers:
            recent_speakers.append(name)
        if len(recent_speakers) >= 3:
            break
    for speaker_name in recent_speakers:
        if speaker_name not in allowed_names:
            allowed_names.append(speaker_name)
    if not allowed_names and agent_order:
        allowed_names = [agent_order[0]]
    allowed_agents = [agent_map[name] for name in agent_order if name in allowed_names]
    if not allowed_agents and agent_order:
        allowed_agents = [agent_map[agent_order[0]]]
    allowed_names_ordered = [agent.name for agent in allowed_agents]
    allowed_name_set = set(allowed_names_ordered)
    visible_limit = min(MAX_BOT_REPLIES_PER_TURN, MAX_GROUP_ROUNDS)
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
    target_rounds = max(2, 1 + len(allowed_agents) * MAX_REPLIES_PER_ROLE)
    max_round_cap = min(MAX_BOT_REPLIES_PER_TURN + 1, original_max_round)
    groupchat.max_round = min(max_round_cap, target_rounds)
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
    cutoff_index = start
    reply_count = 0
    per_role_counts: Dict[str, int] = {}
    messages = manager.groupchat.messages
    for idx in range(start + 1, len(messages)):
        message = messages[idx]
        name = message.get("name")
        if not name or name == user_name:
            continue
        per_role_counts.setdefault(name, 0)
        if per_role_counts[name] >= MAX_REPLIES_PER_ROLE:
            cutoff_index = idx
            continue
        segments = split_content(message.get("content", ""))
        if not segments:
            continue
        payload = "\n".join(segments)
        replies.append({"name": name, "content": payload})
        if name not in allowed_name_set:
            allowed_name_set.add(name)
        per_role_counts[name] += 1
        reply_count += 1
        cutoff_index = idx
        if reply_count >= visible_limit:
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
    if replies:
        ordered = session.get("allowed_names", []) or []
        seen = set(ordered)
        extras: List[str] = []
        for reply in replies:
            name = reply.get("name")
            if name and name not in seen:
                seen.add(name)
                extras.append(name)
        if extras:
            session["allowed_names"] = ordered + extras
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
    allowed_agents, initial_allowed_names, visible_limit = _determine_allowed_agents(
        session, manager, message or ""
    )
    allowed_name_set = set(initial_allowed_names)
    cleanup = _configure_groupchat(manager, user, allowed_agents)

    def run_chat() -> None:
        try:
            user.initiate_chat(manager, message=message, clear_history=False)
        finally:
            pass

    thread = threading.Thread(target=run_chat)
    thread.start()

    delivered_order: List[str] = []
    per_role_counts: Dict[str, int] = {}

    def generate():
        nonlocal idx
        reply_count = 0
        thread_joined = False
        try:
            while True:
                messages = manager.groupchat.messages
                progressed = False
                while idx < len(messages):
                    current = messages[idx]
                    idx += 1
                    name = current.get("name")
                    if not name or name == user.name:
                        continue
                    per_role_counts.setdefault(name, 0)
                    if per_role_counts[name] >= MAX_REPLIES_PER_ROLE:
                        progressed = True
                        continue
                    allowed_name_set.add(name)
                    segments = split_content(current.get("content", ""))
                    if not segments:
                        continue
                    payload = "\n".join(segments)
                    data = json.dumps({"name": name, "content": payload}, ensure_ascii=False)
                    per_role_counts[name] += 1
                    yield f"data: {data}\n\n"
                    reply_count += 1
                    if name not in delivered_order:
                        delivered_order.append(name)
                    progressed = True

                if reply_count >= visible_limit:
                    break
                if thread.is_alive():
                    time.sleep(0.05)
                    continue
                if not thread_joined:
                    thread.join(timeout=0.2)
                    thread_joined = True
                    continue
                if not progressed and idx >= len(manager.groupchat.messages):
                    break
        finally:
            if not thread_joined:
                thread.join(timeout=0.2)
            last_index = min(idx, len(manager.groupchat.messages))
            session["last"] = last_index
            if delivered_order:
                ordered = session.get("allowed_names", []) or []
                seen = set(ordered)
                extras: List[str] = []
                for name in delivered_order:
                    if name not in seen:
                        seen.add(name)
                        extras.append(name)
                if extras:
                    session["allowed_names"] = ordered + extras
            cleanup()

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
