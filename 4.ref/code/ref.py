"""Streamlit chatbot for MOEF-style Q&A with optional RAG and OpenAI web search."""

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
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------------------------
# Logging (WARNING/ERROR only; quiet HTTP stacks)
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    """Configure file and console loggers; suppress INFO and noisy HTTP loggers.

    Returns:
        logging.Logger: Application logger for WARNING/ERROR records.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
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

    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("mof_chatbot")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""

WEB_SEARCH_INSTRUCTIONS = """당신은 최신 정보를 웹 검색 도구로 확인한 뒤 답변하는 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요.
- 서술형 완전 문장, 존댓말.
- 구분선(---, ===, ___) 금지. 취소선(~~) 금지.
- 참조 표시, 출처 나열, URL, '출처:', '참고:' 같은 문구 금지.
"""


def remove_separators(text: str) -> str:
    """Remove markdown strikethrough, horizontal rules, and extra blank lines.

    Args:
        text: Raw assistant or user text.

    Returns:
        str: Cleaned text safe for display.
    """
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    """Return a LangChain chat model for the given provider name.

    Args:
        model_name: One of gpt-4o-mini, gemini-3-pro-preview, claude-sonnet-4-5.
        temperature: Sampling temperature.

    Returns:
        Any: A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the model name is unknown or required API keys are missing.
    """
    if model_name == "gpt-4o-mini":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=key)

    if model_name == "claude-sonnet-4-5":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("Claude 사용을 위해 ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(
            model="claude-sonnet-4-5",
            temperature=temperature,
            api_key=key,
        )

    if model_name == "gemini-3-pro-preview":
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key:
            raise ValueError("Gemini 사용을 위해 GOOGLE_API_KEY가 필요합니다.")
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=temperature,
            google_api_key=key,
        )

    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def _extract_responses_output_text(response: Any) -> str:
    """Extract plain text from OpenAI Responses API result.

    Args:
        response: SDK response object from ``responses.create``.

    Returns:
        str: Combined assistant-visible text.
    """
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", None) or []:
                btype = getattr(block, "type", None)
                if btype == "output_text":
                    parts.append(getattr(block, "text", "") or "")
                elif btype == "text":
                    parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if p).strip()


def run_openai_web_search(
    user_text: str,
    *,
    previous_response_id: str | None,
    openai_client: OpenAI,
) -> tuple[str, str | None]:
    """Run OpenAI Responses API with built-in ``web_search`` tool.

    Args:
        user_text: Latest user message.
        previous_response_id: Prior response id for multi-turn chaining, if any.
        openai_client: Configured OpenAI client.

    Returns:
        tuple[str, str | None]: Assistant text and new ``response.id`` (if present).
    """
    kwargs: dict[str, Any] = {
        "model": "gpt-5-nano",
        "instructions": WEB_SEARCH_INSTRUCTIONS,
        "input": user_text,
        "tools": [{"type": "web_search"}],
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    if not hasattr(openai_client, "responses"):
        raise RuntimeError(
            "설치된 openai 패키지에 Responses API(responses)가 없습니다. "
            "openai SDK를 최신 1.x로 올려 주세요."
        )

    resp = openai_client.responses.create(**kwargs)
    text = _extract_responses_output_text(resp)
    new_id = getattr(resp, "id", None)
    return text, str(new_id) if new_id else None


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    """Format recent chat turns for RAG context.

    Args:
        messages: List of dicts with keys ``role`` and ``content``.
        max_items: Maximum messages to include (counting user+assistant).

    Returns:
        str: Compact dialogue string.
    """
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
    """Build LangChain messages for RAG answering.

    Args:
        question: User question.
        context: Retrieved document excerpts.
        memory_text: Prior dialogue summary string.

    Returns:
        list: Messages for the chat model.
    """
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _append_followup_questions(
    *,
    preferred_llm: Any | None,
    user_q: str,
    answer_without_follow: str,
    openai_key: str,
) -> str:
    """Build follow-up question section using the preferred model or GPT-4o-mini.

    Args:
        preferred_llm: Primary chat model, if already constructed.
        user_q: User question.
        answer_without_follow: Assistant answer before follow-up block.
        openai_key: OpenAI API key for fallback mini model.

    Returns:
        str: Markdown suffix (may be empty if generation fails or no key).
    """
    if preferred_llm is not None:
        extra = _generate_followup_section(preferred_llm, user_q, answer_without_follow)
        if extra.strip():
            return extra
    if openai_key:
        mini = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_key,
        )
        return _generate_followup_section(mini, user_q, answer_without_follow)
    return ""


def _generate_followup_section(llm: Any, user_q: str, answer: str) -> str:
    """Create markdown block with three suggested follow-up questions.

    Args:
        llm: Chat model.
        user_q: Original user question.
        answer: Assistant answer (may be truncated internally).

    Returns:
        str: Markdown section to append after the main answer.
    """
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


def _process_pdf_uploads(uploaded_files: list[Any]) -> FAISS | None:
    """Load PDFs, split, embed in batches, and build a FAISS index.

    Args:
        uploaded_files: Streamlit uploaded file objects.

    Returns:
        FAISS | None: Vector store or None if nothing processed.
    """
    if not uploaded_files:
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PDF 임베딩에 OPENAI_API_KEY가 필요합니다.")

    all_docs: list[Any] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    batch_size = 30
    store: FAISS | None = None
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        part = FAISS.from_documents(batch, embeddings)
        if store is None:
            store = part
        else:
            store.merge_from(part)
    return store


def _init_session() -> None:
    """Initialize ``st.session_state`` keys used by the app."""
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "vectorstore": None,
        "processed_names": [],
        "web_search_last_response_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    """Run the Streamlit application entrypoint."""
    st.set_page_config(
        page_title="숭실대학교 챗봇",
        page_icon="📚",
        layout="wide",
    )
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
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    with st.sidebar:
        model_choice = st.radio(
            "LLM 모델 선택",
            ("gpt-4o-mini", "gemini-3-pro-preview", "claude-sonnet-4-5"),
            index=0,
        )
        internet_choice = st.radio(
            "인터넷 검색 선택",
            ("사용 안 함", "OpenAI 웹 검색"),
            index=0,
        )
        rag_choice = st.radio(
            "RAG (PDF 검색) 선택",
            ("사용 안 함", "RAG 사용"),
            index=0,
        )

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    vs = _process_pdf_uploads(list(uploads))
                    st.session_state.vectorstore = vs
                    st.session_state.processed_names = [u.name for u in uploads]
                    st.success("PDF 처리가 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류가 발생했습니다: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        if st.button("대화 초기화"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.web_search_last_response_id = None
            st.rerun()

        mem_count = len(st.session_state.conversation_memory)
        vs = st.session_state.vectorstore
        file_count = len(st.session_state.processed_names)
        settings_text = (
            f"모델: {model_choice}\n"
            f"인터넷 검색: {internet_choice}\n"
            f"RAG: {rag_choice}\n"
            f"처리된 PDF 파일 수: {file_count}\n"
            f"벡터 스토어: {'있음' if vs is not None else '없음'}\n"
            f"대화 기록(메시지) 수: {mem_count}"
        )
        st.text(settings_text)

    if internet_choice != "OpenAI 웹 검색 (GPT-5 nano)":
        st.session_state.web_search_last_response_id = None

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = remove_separators(msg["content"])
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[
            -50:
        ]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        preferred_follow_llm: Any | None = None

        try:
            if internet_choice == "OpenAI 웹 검색 (GPT-5 nano)":
                if not openai_key:
                    full_answer = (
                        "# 안내\n\n"
                        "OpenAI 웹 검색을 사용하려면 `.env`에 **OPENAI_API_KEY**를 설정해 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                    preferred_follow_llm = None
                else:
                    client = OpenAI(api_key=openai_key)
                    prev = st.session_state.web_search_last_response_id
                    text, new_id = run_openai_web_search(
                        user_input,
                        previous_response_id=prev,
                        openai_client=client,
                    )
                    st.session_state.web_search_last_response_id = new_id
                    full_answer = remove_separators(text or "(응답이 비어 있습니다.)")
                    placeholder.markdown(full_answer)
                    preferred_follow_llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0.3,
                        api_key=openai_key,
                    )

            elif rag_choice == "RAG 사용":
                vs = st.session_state.vectorstore
                if vs is None:
                    full_answer = (
                        "# 안내\n\n"
                        "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                else:
                    llm = get_llm(model_choice)
                    preferred_follow_llm = llm
                    mem_txt = _format_memory_block(
                        st.session_state.conversation_memory[:-1]
                    )
                    docs = vs.similarity_search(user_input, k=10)
                    context = "\n\n".join(d.page_content for d in docs)
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    stream_iter = llm.stream(messages)
                    acc = ""
                    for chunk in stream_iter:
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)

            else:
                llm = get_llm(model_choice)
                preferred_follow_llm = llm
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1]
                )
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [
                    SystemMessage(content=sys),
                    HumanMessage(content=user_input),
                ]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

            if not full_answer.lstrip().startswith("# 오류"):
                base_for_follow = full_answer
                follow = _append_followup_questions(
                    preferred_llm=preferred_follow_llm,
                    user_q=user_input,
                    answer_without_follow=base_for_follow,
                    openai_key=openai_key,
                )
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = (
                f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            )
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_answer}
        )
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer}
        )
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[
                -50:
            ]


if __name__ == "__main__":
    main()
