"""멀티유저 RAG 챗봇 — Supabase 사용자·세션·벡터 저장 + Streamlit UI."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_BATCH_SIZE = 10
CHAT_MEMORY_LIMIT = 50
PBKDF2_ITERATIONS = 100_000

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""

CUSTOM_CSS = """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
"""


def get_config_value(key: str) -> str | None:
    """st.secrets 우선, 없으면 .env / os.getenv."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:  # noqa: BLE001
        pass
    val = (os.getenv(key) or "").strip()
    return val or None


def _setup_logging() -> logging.Logger:
    """Streamlit Cloud 등 쓰기 불가 환경에서도 앱이 기동되도록 로깅 경로를 탐색한다."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    candidates = [
        LOG_DIR,
        Path(tempfile.gettempdir()) / "multiusers_logs",
    ]

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            fh = logging.FileHandler(base / log_name, encoding="utf-8")
            fh.setLevel(logging.WARNING)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            break
        except (PermissionError, OSError):
            continue

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Password helpers (PBKDF2-SHA256)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split(":", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return secrets.compare_digest(digest.hex(), expected_hex)


# ---------------------------------------------------------------------------
# Text / LLM helpers
# ---------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


def get_llm(api_key: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=api_key)


def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


def _format_memory_block(messages: list[dict[str, str]], max_items: int = CHAT_MEMORY_LIMIT) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""

    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 요약해 대화 세션 제목을 한국어로 20자 이내로 작성하세요. "
        "제목만 출력하고 따옴표는 쓰지 마세요.\n\n"
        f"[질문]\n{first_q}\n\n[답변]\n{first_a[:800]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = remove_separators(str(getattr(out, "content", "") or "")).strip()
        return title[:40] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Session title generation failed: %s", exc)
        return "새 세션"


def _first_qa_pair(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    user_text = ""
    for m in messages:
        if m["role"] == "user" and not user_text:
            user_text = m["content"]
        elif m["role"] == "assistant" and user_text:
            return user_text, m["content"]
    return None


# ---------------------------------------------------------------------------
# Auth data layer
# ---------------------------------------------------------------------------
def register_user(sb: Client, login_id: str, password: str) -> int:
    login_id = login_id.strip()
    if not login_id:
        raise ValueError("아이디를 입력해 주세요.")

    existing = sb.table("user").select("id").eq("login_id", login_id).limit(1).execute()
    if existing.data:
        raise ValueError("이미 사용 중인 아이디입니다.")

    resp = (
        sb.table("user")
        .insert({"login_id": login_id, "password_hash": hash_password(password)})
        .execute()
    )
    if not resp.data:
        raise RuntimeError("회원가입에 실패했습니다.")
    return int(resp.data[0]["id"])


def authenticate_user(sb: Client, login_id: str, password: str) -> dict[str, Any]:
    login_id = login_id.strip()
    resp = (
        sb.table("user")
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise ValueError("아이디 또는 비밀번호가 올바르지 않습니다.")

    row = resp.data[0]
    if not verify_password(password, row["password_hash"]):
        raise ValueError("아이디 또는 비밀번호가 올바르지 않습니다.")

    return {"id": int(row["id"]), "login_id": row["login_id"]}


# ---------------------------------------------------------------------------
# Supabase data layer (user-scoped)
# ---------------------------------------------------------------------------
def fetch_sessions(sb: Client, user_id: int) -> list[dict[str, Any]]:
    resp = (
        sb.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def create_session_row(sb: Client, user_id: int, title: str) -> str:
    resp = sb.table("chat_sessions").insert({"user_id": user_id, "title": title}).execute()
    if not resp.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return str(resp.data[0]["id"])


def update_session_title(sb: Client, user_id: int, session_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).eq("user_id", user_id).execute()


def delete_session_row(sb: Client, user_id: int, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def save_messages(
    sb: Client,
    user_id: int,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
    if not messages:
        return
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
        }
        for m in messages
    ]
    sb.table("chat_messages").insert(rows).execute()


def load_messages(sb: Client, user_id: int, session_id: str) -> list[dict[str, str]]:
    resp = (
        sb.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in (resp.data or [])]


def get_vector_file_names(sb: Client, user_id: int, session_id: str) -> list[str]:
    resp = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})


def insert_vector_chunks(
    sb: Client,
    user_id: int,
    session_id: str,
    file_name: str,
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
) -> int:
    texts = [c.page_content for c in chunks]
    if not texts:
        return 0

    vectors = embeddings.embed_documents(texts)
    rows: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "file_name": file_name,
                "content": chunk.page_content,
                "metadata": chunk.metadata or {},
                "embedding": vector,
            }
        )

    inserted = 0
    for i in range(0, len(rows), VECTOR_BATCH_SIZE):
        batch = rows[i : i + VECTOR_BATCH_SIZE]
        sb.table("vector_documents").insert(batch).execute()
        inserted += len(batch)
    return inserted


def copy_vectors_to_session(
    sb: Client,
    user_id: int,
    source_session_id: str,
    target_session_id: str,
) -> None:
    resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", source_session_id)
        .eq("user_id", user_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return

    new_rows = [
        {
            "user_id": user_id,
            "session_id": target_session_id,
            "file_name": r["file_name"],
            "content": r["content"],
            "metadata": r.get("metadata") or {},
            "embedding": r["embedding"],
        }
        for r in rows
    ]
    for i in range(0, len(new_rows), VECTOR_BATCH_SIZE):
        sb.table("vector_documents").insert(new_rows[i : i + VECTOR_BATCH_SIZE]).execute()


def search_vectors_rpc(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    user_id: int,
    session_id: str,
    query: str,
    k: int = 10,
) -> list[Document]:
    query_vec = embeddings.embed_query(query)
    try:
        resp = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_vec,
                "match_count": k,
                "filter_session_id": session_id,
                "filter_user_id": user_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in resp.data or []:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "file_name": row.get("file_name"),
                        "session_id": row.get("session_id"),
                        "similarity": row.get("similarity"),
                        **(row.get("metadata") or {}),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC vector search failed, fallback: %s", exc)
        return _search_vectors_fallback(sb, user_id, session_id, query, k)


def _search_vectors_fallback(
    sb: Client,
    user_id: int,
    session_id: str,
    query: str,
    k: int,
) -> list[Document]:
    resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .limit(200)
        .execute()
    )
    rows = resp.data or []
    q = query.lower()
    scored: list[tuple[int, Document]] = []
    for row in rows:
        content = row.get("content", "")
        score = content.lower().count(q) if q else 0
        scored.append(
            (
                score,
                Document(
                    page_content=content,
                    metadata={"file_name": row.get("file_name"), **(row.get("metadata") or {})},
                ),
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
        "logged_in_user_id": None,
        "logged_in_login_id": None,
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "current_session_id": None,
        "sessions_cache": [],
        "selected_session_id": None,
        "vectordb_panel": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def refresh_sessions(sb: Client, user_id: int) -> None:
    st.session_state.sessions_cache = fetch_sessions(sb, user_id)


def _session_label_map(sessions: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for s in sessions:
        sid = str(s["id"])
        title = s.get("title") or "새 세션"
        updated = (s.get("updated_at") or s.get("created_at") or "")[:16]
        labels[sid] = f"{title} ({updated})"
    return labels


def load_session_into_state(sb: Client, user_id: int, session_id: str) -> None:
    messages = load_messages(sb, user_id, session_id)
    file_names = get_vector_file_names(sb, user_id, session_id)
    st.session_state.current_session_id = session_id
    st.session_state.selected_session_id = session_id
    st.session_state.chat_history = messages
    st.session_state.conversation_memory = messages[-CHAT_MEMORY_LIMIT:]
    st.session_state.processed_names = file_names


def clear_screen_state() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.current_session_id = None
    st.session_state.selected_session_id = None
    st.session_state.vectordb_panel = ""


def logout_user() -> None:
    st.session_state.logged_in_user_id = None
    st.session_state.logged_in_login_id = None
    clear_screen_state()
    st.session_state.sessions_cache = []


def ensure_working_session(sb: Client, user_id: int, title: str = "새 세션") -> str:
    if st.session_state.current_session_id:
        return st.session_state.current_session_id
    sid = create_session_row(sb, user_id, title)
    st.session_state.current_session_id = sid
    refresh_sessions(sb, user_id)
    return sid


def auto_save_session(sb: Client, user_id: int, llm: ChatOpenAI | None = None) -> None:
    messages = st.session_state.chat_history
    if not messages:
        return

    session_id = ensure_working_session(sb, user_id)
    save_messages(sb, user_id, session_id, messages)

    pair = _first_qa_pair(messages)
    if pair and llm is not None:
        title = generate_session_title(llm, pair[0], pair[1])
        update_session_title(sb, user_id, session_id, title)

    refresh_sessions(sb, user_id)


def manual_save_new_session(sb: Client, user_id: int, llm: ChatOpenAI) -> str | None:
    messages = st.session_state.chat_history
    if not messages:
        st.sidebar.warning("저장할 대화가 없습니다.")
        return None

    pair = _first_qa_pair(messages)
    title = generate_session_title(llm, pair[0], pair[1]) if pair else "새 세션"

    new_id = create_session_row(sb, user_id, title)
    save_messages(sb, user_id, new_id, messages)

    old_id = st.session_state.current_session_id
    if old_id and old_id != new_id:
        copy_vectors_to_session(sb, user_id, old_id, new_id)

    st.session_state.current_session_id = new_id
    st.session_state.selected_session_id = new_id
    refresh_sessions(sb, user_id)
    return new_id


def _process_pdf_uploads(
    sb: Client,
    user_id: int,
    session_id: str,
    uploaded_files: list[Any],
    api_key: str,
) -> list[str]:
    if not uploaded_files:
        return []

    embeddings = get_embeddings(api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed: list[str] = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if not docs:
                continue
            for doc in docs:
                doc.metadata = {**(doc.metadata or {}), "file_name": uf.name, "source": uf.name}
            chunks = splitter.split_documents(docs)
            if not chunks:
                continue
            insert_vector_chunks(sb, user_id, session_id, uf.name, chunks, embeddings)
            processed.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_header(title_html: str) -> None:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(title_html, unsafe_allow_html=True)
    with c3:
        st.empty()


def render_login_screen(sb: Client) -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_header(
        """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
<p style="text-align:center; color:#666;">로그인 또는 회원가입 후 이용해 주세요.</p>
"""
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

        with tab_login:
            login_id = st.text_input("아이디", key="login_id_input")
            password = st.text_input("비밀번호", type="password", key="login_pw_input")
            if st.button("로그인", use_container_width=True, key="btn_login"):
                try:
                    user = authenticate_user(sb, login_id, password)
                    st.session_state.logged_in_user_id = user["id"]
                    st.session_state.logged_in_login_id = user["login_id"]
                    clear_screen_state()
                    refresh_sessions(sb, user["id"])
                    st.success(f"{user['login_id']}님, 환영합니다!")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        with tab_signup:
            new_id = st.text_input("새 아이디", key="signup_id_input")
            new_pw = st.text_input("비밀번호", type="password", key="signup_pw_input")
            if st.button("회원가입", use_container_width=True, key="btn_signup"):
                try:
                    uid = register_user(sb, new_id, new_pw)
                    st.session_state.logged_in_user_id = uid
                    st.session_state.logged_in_login_id = new_id.strip()
                    clear_screen_state()
                    refresh_sessions(sb, uid)
                    st.success("회원가입이 완료되었습니다. 자동 로그인되었습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))


def on_session_select() -> None:
    user_id = st.session_state.get("logged_in_user_id")
    if not user_id:
        return
    url = get_config_value("SUPABASE_URL")
    key = get_config_value("SUPABASE_ANON_KEY")
    if not url or not key:
        return
    sb = get_supabase_client(url, key)
    selected = st.session_state.get("session_selector")
    if selected:
        load_session_into_state(sb, user_id, selected)


def render_dashboard(sb: Client, openai_key: str) -> None:
    user_id = st.session_state.logged_in_user_id
    login_id = st.session_state.logged_in_login_id or ""
    llm = get_llm(openai_key)

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header(
        """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
"""
    )

    if not st.session_state.sessions_cache:
        refresh_sessions(sb, user_id)

    with st.sidebar:
        st.markdown(f"**로그인:** `{login_id}`")
        if st.button("로그아웃", use_container_width=True):
            logout_user()
            st.rerun()

        st.divider()
        st.markdown("**LLM 모델**")
        st.text(MODEL_NAME)

        st.divider()
        st.markdown("**세션 관리**")

        sessions = st.session_state.sessions_cache
        label_map = _session_label_map(sessions)
        options = [str(s["id"]) for s in sessions]
        if not options:
            st.caption("저장된 세션이 없습니다.")

        current_sel = st.session_state.selected_session_id
        index = options.index(current_sel) if current_sel in options else 0 if options else 0

        if options:
            st.selectbox(
                "세션 선택",
                options,
                index=index,
                format_func=lambda sid: label_map.get(sid, sid),
                key="session_selector",
                on_change=on_session_select,
            )

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("세션저장", use_container_width=True):
                try:
                    new_id = manual_save_new_session(sb, user_id, llm)
                    if new_id:
                        st.success("새 세션이 저장되었습니다.")
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 저장 실패: {exc}")

        with bc2:
            if st.button("세션로드", use_container_width=True):
                selected = st.session_state.get("session_selector")
                if not selected:
                    st.warning("로드할 세션을 선택해 주세요.")
                else:
                    try:
                        load_session_into_state(sb, user_id, selected)
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 로드 실패: {exc}")

        bc3, bc4 = st.columns(2)
        with bc3:
            if st.button("세션삭제", use_container_width=True):
                target = st.session_state.get("session_selector") or st.session_state.current_session_id
                if not target:
                    st.warning("삭제할 세션을 선택해 주세요.")
                else:
                    try:
                        delete_session_row(sb, user_id, target)
                        if st.session_state.current_session_id == target:
                            clear_screen_state()
                        refresh_sessions(sb, user_id)
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")

        with bc4:
            if st.button("화면초기화", use_container_width=True):
                clear_screen_state()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            sid = st.session_state.current_session_id or st.session_state.get("session_selector")
            if not sid:
                st.session_state.vectordb_panel = "현재 활성 세션이 없습니다."
            else:
                names = get_vector_file_names(sb, user_id, sid)
                st.session_state.vectordb_panel = (
                    "\n".join(f"- {n}" for n in names)
                    if names
                    else "벡터 DB에 저장된 파일이 없습니다."
                )

        if st.session_state.vectordb_panel:
            st.markdown("**Vector DB 파일 목록**")
            st.text(st.session_state.vectordb_panel)

        st.divider()
        rag_choice = st.radio("RAG (PDF 검색) 선택", ("사용 안 함", "RAG 사용"), index=1)

        uploads = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    sid = ensure_working_session(sb, user_id)
                    names = _process_pdf_uploads(sb, user_id, sid, list(uploads), openai_key)
                    existing = set(st.session_state.processed_names)
                    st.session_state.processed_names = sorted(existing | set(names))
                    auto_save_session(sb, user_id, llm)
                    st.success(f"PDF {len(names)}개 처리·저장 완료")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        sid = st.session_state.current_session_id
        st.text(
            f"현재 세션 ID: {sid or '(없음)'}\n"
            f"대화 메시지 수: {len(st.session_state.chat_history)}\n"
            f"저장된 세션 수: {len(st.session_state.sessions_cache)}"
        )

    st.markdown(
        f"### 안녕하세요, **{login_id}**님!\n"
        "PDF를 업로드하고 질문해 보세요. 대화는 자동으로 저장됩니다."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > CHAT_MEMORY_LIMIT:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-CHAT_MEMORY_LIMIT:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            if rag_choice == "RAG 사용":
                sid = st.session_state.current_session_id or ensure_working_session(sb, user_id)
                embeddings = get_embeddings(openai_key)
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = search_vectors_rpc(sb, embeddings, user_id, sid, user_input, k=10)

                if not docs and not st.session_state.processed_names:
                    full_answer = (
                        "# 안내\n\n"
                        "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                else:
                    context = "\n\n".join(d.page_content for d in docs) if docs else "(관련 문서 없음)"
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    acc = ""
                    for chunk in llm.stream(messages):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

            if full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
    if len(st.session_state.conversation_memory) > CHAT_MEMORY_LIMIT:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-CHAT_MEMORY_LIMIT:]

    try:
        auto_save_session(sb, user_id, llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 세션 저장 실패: %s", exc)
        st.sidebar.warning(f"자동 저장 실패: {exc}")


def main() -> None:
    st.set_page_config(page_title="숭실대학교 RAG 챗봇", page_icon="📚", layout="wide")
    _init_session()

    openai_key = get_config_value("OPENAI_API_KEY")
    supabase_url = get_config_value("SUPABASE_URL")
    supabase_key = get_config_value("SUPABASE_ANON_KEY")

    missing = [
        name
        for name, val in {
            "OPENAI_API_KEY": openai_key,
            "SUPABASE_URL": supabase_url,
            "SUPABASE_ANON_KEY": supabase_key,
        }.items()
        if not val
    ]
    if missing:
        st.error(
            "다음 환경 변수가 설정되지 않았습니다: "
            + ", ".join(missing)
            + f"\n\n로컬: `{ENV_PATH}` · Streamlit Cloud: Secrets 탭을 확인해 주세요."
        )
        return

    sb = get_supabase_client(supabase_url, supabase_key)

    if not st.session_state.logged_in_user_id:
        render_login_screen(sb)
        return

    render_dashboard(sb, openai_key)


if __name__ == "__main__":
    main()
