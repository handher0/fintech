"""멀티세션 RAG 챗봇 — Supabase 세션·벡터 저장 + Streamlit UI."""

from __future__ import annotations

import logging
import os
import re
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

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"multiref_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = LOG_DIR / log_name

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiref")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_config() -> dict[str, str | None]:
    return {
        "openai": (os.getenv("OPENAI_API_KEY") or "").strip() or None,
        "supabase_url": (os.getenv("SUPABASE_URL") or "").strip() or None,
        "supabase_key": (os.getenv("SUPABASE_ANON_KEY") or "").strip() or None,
    }


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
# Supabase data layer
# ---------------------------------------------------------------------------
def fetch_sessions(sb: Client) -> list[dict[str, Any]]:
    resp = (
        sb.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def create_session_row(sb: Client, title: str) -> str:
    resp = sb.table("chat_sessions").insert({"title": title}).execute()
    if not resp.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return str(resp.data[0]["id"])


def update_session_title(sb: Client, session_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def delete_session_row(sb: Client, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).execute()


def save_messages(sb: Client, session_id: str, messages: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    if not messages:
        return
    rows = [
        {"session_id": session_id, "role": m["role"], "content": m["content"]}
        for m in messages
    ]
    sb.table("chat_messages").insert(rows).execute()


def load_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    resp = (
        sb.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in (resp.data or [])]


def get_vector_file_names(sb: Client, session_id: str) -> list[str]:
    resp = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})
    return names


def insert_vector_chunks(
    sb: Client,
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


def copy_vectors_to_session(sb: Client, source_session_id: str, target_session_id: str) -> None:
    resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return

    new_rows = [
        {
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
        logger.warning("RPC vector search failed, fallback to client filter: %s", exc)
        return _search_vectors_fallback(sb, session_id, query, k)


def _search_vectors_fallback(
    sb: Client,
    session_id: str,
    query: str,
    k: int,
) -> list[Document]:
    resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata")
        .eq("session_id", session_id)
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
# Session state orchestration
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
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


def refresh_sessions(sb: Client) -> None:
    st.session_state.sessions_cache = fetch_sessions(sb)


def _session_label_map(sessions: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for s in sessions:
        sid = str(s["id"])
        title = s.get("title") or "새 세션"
        updated = (s.get("updated_at") or s.get("created_at") or "")[:16]
        labels[sid] = f"{title} ({updated})"
    return labels


def load_session_into_state(sb: Client, session_id: str) -> None:
    messages = load_messages(sb, session_id)
    file_names = get_vector_file_names(sb, session_id)
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


def ensure_working_session(sb: Client, title: str = "새 세션") -> str:
    if st.session_state.current_session_id:
        return st.session_state.current_session_id
    sid = create_session_row(sb, title)
    st.session_state.current_session_id = sid
    refresh_sessions(sb)
    return sid


def auto_save_session(sb: Client, llm: ChatOpenAI | None = None) -> None:
    messages = st.session_state.chat_history
    if not messages:
        return

    session_id = ensure_working_session(sb)
    save_messages(sb, session_id, messages)

    pair = _first_qa_pair(messages)
    if pair and llm is not None:
        first_q, first_a = pair
        title = generate_session_title(llm, first_q, first_a)
        update_session_title(sb, session_id, title)

    refresh_sessions(sb)


def manual_save_new_session(sb: Client, llm: ChatOpenAI) -> str | None:
    messages = st.session_state.chat_history
    if not messages:
        st.sidebar.warning("저장할 대화가 없습니다.")
        return None

    pair = _first_qa_pair(messages)
    if pair:
        title = generate_session_title(llm, pair[0], pair[1])
    else:
        title = "새 세션"

    new_id = create_session_row(sb, title)
    save_messages(sb, new_id, messages)

    old_id = st.session_state.current_session_id
    if old_id and old_id != new_id:
        copy_vectors_to_session(sb, old_id, new_id)

    st.session_state.current_session_id = new_id
    st.session_state.selected_session_id = new_id
    refresh_sessions(sb)
    return new_id


def _process_pdf_uploads(
    sb: Client,
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
            insert_vector_chunks(sb, session_id, uf.name, chunks, embeddings)
            processed.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------
def on_session_select() -> None:
    cfg = get_config()
    if not cfg["supabase_url"] or not cfg["supabase_key"]:
        return
    sb = get_supabase_client(cfg["supabase_url"], cfg["supabase_key"])
    selected = st.session_state.get("session_selector")
    if selected:
        load_session_into_state(sb, selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">멀티세션</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    cfg = get_config()
    missing = [k for k, v in {
        "OPENAI_API_KEY": cfg["openai"],
        "SUPABASE_URL": cfg["supabase_url"],
        "SUPABASE_ANON_KEY": cfg["supabase_key"],
    }.items() if not v]

    if missing:
        st.error(
            "다음 환경 변수가 설정되지 않았습니다: "
            + ", ".join(missing)
            + f"\n\n`{ENV_PATH}` 파일을 확인해 주세요."
        )
        return

    sb = get_supabase_client(cfg["supabase_url"], cfg["supabase_key"])
    if not st.session_state.sessions_cache:
        refresh_sessions(sb)

    llm = get_llm(cfg["openai"])

    with st.sidebar:
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
                    new_id = manual_save_new_session(sb, llm)
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
                        load_session_into_state(sb, selected)
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
                        delete_session_row(sb, target)
                        if st.session_state.current_session_id == target:
                            clear_screen_state()
                        refresh_sessions(sb)
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
                names = get_vector_file_names(sb, sid)
                if names:
                    st.session_state.vectordb_panel = "\n".join(f"- {n}" for n in names)
                else:
                    st.session_state.vectordb_panel = "벡터 DB에 저장된 파일이 없습니다."

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
                    sid = ensure_working_session(sb)
                    names = _process_pdf_uploads(sb, sid, list(uploads), cfg["openai"])
                    existing = set(st.session_state.processed_names)
                    st.session_state.processed_names = sorted(existing | set(names))
                    auto_save_session(sb, llm)
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
                sid = st.session_state.current_session_id or ensure_working_session(sb)
                embeddings = get_embeddings(cfg["openai"])
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = search_vectors_rpc(sb, embeddings, sid, user_input, k=10)

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
        auto_save_session(sb, llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 세션 저장 실패: %s", exc)
        st.sidebar.warning(f"자동 저장 실패: {exc}")


if __name__ == "__main__":
    main()
