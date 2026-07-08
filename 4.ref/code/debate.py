"""Two Agent Debate — 2개의 AI Agent가 주제에 대해 토론하는 Streamlit 앱."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ---------------------------------------------------------------------------
# 경로 & 환경
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"

load_dotenv(dotenv_path=ENV_PATH)

STOP_MARKER = "[토론종료]"

WEB_SEARCH_INSTRUCTIONS = """당신은 최신 정보를 웹 검색 도구로 확인한 뒤 토론에 활용할 근거를 정리하는 조수입니다.
한국어로 핵심 사실과 논거를 간결하게 정리하세요. 출처 URL은 나열하지 마세요."""

DEBATE_STYLE = """답변 규칙:
- 마크다운 헤딩(# ## ###)으로 구조화하세요.
- 서술형 완전 문장, 존댓말.
- 이전 발언과 동일한 내용을 반복하지 마세요.
- 더 이상 새로운 논점이 없으면 응답 맨 첫 줄에 '[토론종료]'만 출력하세요.
- 구분선(---)과 취소선(~~)은 사용하지 마세요."""


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    """마크다운 구분선·취소선을 제거한다."""
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    """사이드바에서 선택한 model_name 그대로 LangChain LLM을 반환한다."""
    if model_name == "gpt-4o-mini":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=key)

    if model_name == "claude-sonnet-4-5":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("Claude 사용을 위해 ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature, api_key=key)

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
    """OpenAI Responses API 응답에서 텍스트를 추출한다."""
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", None) or []:
                btype = getattr(block, "type", None)
                if btype in ("output_text", "text"):
                    parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if p).strip()


def run_web_search(query: str, openai_key: str) -> str:
    """GPT-5 nano + Responses API web_search로 인터넷 검색을 수행한다."""
    client = OpenAI(api_key=openai_key)
    if not hasattr(client, "responses"):
        raise RuntimeError(
            "설치된 openai 패키지에 Responses API가 없습니다. openai SDK 1.x 이상이 필요합니다."
        )
    resp = client.responses.create(
        model="gpt-5-nano",
        instructions=WEB_SEARCH_INSTRUCTIONS,
        input=query,
        tools=[{"type": "web_search"}],
    )
    return _extract_responses_output_text(resp) or "(검색 결과를 가져오지 못했습니다.)"


def build_vectorstore(uploaded_files: list[Any]) -> FAISS | None:
    """업로드된 PDF로 FAISS 벡터 스토어를 만든다."""
    if not uploaded_files:
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("문서검색에 OPENAI_API_KEY가 필요합니다.")

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

    store: FAISS | None = None
    batch_size = 30
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        part = FAISS.from_documents(batch, embeddings)
        store = part if store is None else store.merge_from(part)
    return store


def format_debate_history(turns: list[dict[str, Any]]) -> str:
    """지금까지의 토론 발언을 문자열로 정리한다."""
    lines: list[str] = []
    for t in turns:
        lines.append(f"[{t['round']}회차] {t['agent']}: {t['content']}")
    return "\n\n".join(lines) if lines else "(아직 발언 없음)"


def is_stop_response(text: str) -> bool:
    """에이전트가 더 할 말이 없다고 표시했는지 확인한다."""
    stripped = text.strip()
    return stripped == STOP_MARKER or stripped.startswith(STOP_MARKER)


def stream_llm(llm: Any, messages: list, placeholder) -> str:
    """LLM 응답을 스트리밍으로 placeholder에 표시하고 전체 텍스트를 반환한다."""
    acc = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "") or ""
        if piece:
            acc += piece
            placeholder.markdown(remove_separators(acc) + "▌")
    final = remove_separators(acc)
    placeholder.markdown(final)
    return final


def gather_tool_context(
    tool_type: str,
    *,
    topic: str,
    position: str,
    agent_name: str,
    opponent_last: str,
    vectorstore: FAISS | None,
    openai_key: str,
) -> str:
    """도구 유형에 따라 토론에 활용할 참고 자료를 수집한다."""
    if tool_type == "도구 없음":
        return ""

    query = (
        f"토론 주제: {topic}\n"
        f"내 입장({agent_name}): {position}\n"
        f"상대방 최근 발언: {opponent_last or '(없음)'}\n"
        f"이 입장을 뒷받침할 근거를 찾아 주세요."
    )

    if tool_type == "인터넷 검색":
        if not openai_key:
            return "(OPENAI_API_KEY가 없어 인터넷 검색을 수행할 수 없습니다.)"
        return run_web_search(query, openai_key)

    if tool_type == "문서검색":
        if vectorstore is None:
            return "(업로드된 PDF 문서가 없습니다.)"
        docs = vectorstore.similarity_search(query, k=6)
        return "\n\n".join(d.page_content for d in docs)

    return ""


def build_agent_messages(
    *,
    agent_name: str,
    position: str,
    topic: str,
    history_text: str,
    tool_context: str,
    is_opening: bool,
) -> list[SystemMessage | HumanMessage]:
    """에이전트 1회 발언용 LangChain 메시지를 구성한다."""
    tool_block = f"\n\n[참고 자료]\n{tool_context}" if tool_context else ""
    role_hint = "첫 발언으로 자신의 입장을 명확히 제시하세요." if is_opening else "상대방 발언에 반박하거나 보완하세요."

    system = f"""당신은 토론자 '{agent_name}'입니다.

[토론 주제]
{topic}

[당신의 입장]
{position}

{DEBATE_STYLE}

{role_hint}
{tool_block}

[지금까지의 토론]
{history_text}
"""
    user = "위 조건에 맞게 토론 발언을 작성해 주세요."
    return [SystemMessage(content=system), HumanMessage(content=user)]


def generate_agent_turn(
    *,
    llm: Any,
    agent_name: str,
    position: str,
    topic: str,
    turns: list[dict[str, Any]],
    tool_type: str,
    vectorstore: FAISS | None,
    openai_key: str,
    is_opening: bool,
    display_container,
) -> str:
    """에이전트 1회 발언을 생성한다 (도구 사용 시 스피너 표시)."""
    history_text = format_debate_history(turns)
    opponent_last = turns[-1]["content"] if turns else ""

    tool_context = ""
    if tool_type != "도구 없음":
        with display_container:
            with st.spinner(f"{agent_name}이(가) 도구를 사용하는 중..."):
                tool_context = gather_tool_context(
                    tool_type,
                    topic=topic,
                    position=position,
                    agent_name=agent_name,
                    opponent_last=opponent_last,
                    vectorstore=vectorstore,
                    openai_key=openai_key,
                )

    messages = build_agent_messages(
        agent_name=agent_name,
        position=position,
        topic=topic,
        history_text=history_text,
        tool_context=tool_context,
        is_opening=is_opening,
    )

    placeholder = display_container.empty()
    with display_container:
        with st.spinner(f"{agent_name}이(가) 답변을 생성하는 중..."):
            return stream_llm(llm, messages, placeholder)


def run_moderator_summary(
    *,
    llm: Any,
    topic: str,
    agent1_name: str,
    agent1_position: str,
    agent2_name: str,
    agent2_position: str,
    turns: list[dict[str, Any]],
    display_container,
) -> str:
    """사회자가 토론을 정리하고 개인 의견을 제시한다."""
    history_text = format_debate_history(turns)
    system = f"""당신은 토론 사회자입니다. 아래 토론을 정리하고 자신의 개인적인 견해를 밝히세요.

[토론 주제]
{topic}

[에이전트1: {agent1_name}] 입장: {agent1_position}
[에이전트2: {agent2_name}] 입장: {agent2_position}

{DEBATE_STYLE}

다음 구조로 작성하세요:
# 토론 요약
## {agent1_name}의 핵심 논거
## {agent2_name}의 핵심 논거
# 사회자 정리 및 개인 의견

[전체 토론 기록]
{history_text}
"""
    messages = [SystemMessage(content=system), HumanMessage(content="토론을 정리해 주세요.")]
    placeholder = display_container.empty()
    with display_container:
        with st.spinner("사회자가 토론을 정리하는 중..."):
            return stream_llm(llm, messages, placeholder)


def run_debate(
    *,
    model_name: str,
    max_rounds: int,
    tool_type: str,
    topic: str,
    agent1_name: str,
    agent1_position: str,
    agent2_name: str,
    agent2_position: str,
    agent1_vs: FAISS | None,
    agent2_vs: FAISS | None,
    progress_bar,
    debate_area,
) -> list[dict[str, Any]]:
    """최대 토론 횟수만큼 양측 에이전트가 번갈아 발언한다."""
    llm = get_llm(model_name, temperature=0.7)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    turns: list[dict[str, Any]] = []
    stopped_early = False

    for round_num in range(1, max_rounds + 1):
        progress_bar.progress(round_num / max_rounds, text=f"토론 진행 중... ({round_num}/{max_rounds}회차)")

        # --- 에이전트 1 ---
        with debate_area:
            st.markdown(f"### 🎤 {round_num}회차 — {agent1_name}")
        content1 = generate_agent_turn(
            llm=llm,
            agent_name=agent1_name,
            position=agent1_position,
            topic=topic,
            turns=turns,
            tool_type=tool_type,
            vectorstore=agent1_vs,
            openai_key=openai_key,
            is_opening=(round_num == 1),
            display_container=debate_area,
        )
        if is_stop_response(content1):
            stopped_early = True
            break
        turns.append({"round": round_num, "agent": agent1_name, "content": content1})

        # --- 에이전트 2 ---
        with debate_area:
            st.markdown(f"### 🎤 {round_num}회차 — {agent2_name}")
        content2 = generate_agent_turn(
            llm=llm,
            agent_name=agent2_name,
            position=agent2_position,
            topic=topic,
            turns=turns,
            tool_type=tool_type,
            vectorstore=agent2_vs,
            openai_key=openai_key,
            is_opening=False,
            display_container=debate_area,
        )
        if is_stop_response(content2):
            stopped_early = True
            break
        turns.append({"round": round_num, "agent": agent2_name, "content": content2})

    progress_bar.progress(1.0, text="토론 완료" if not stopped_early else "토론 조기 종료")

    # --- 사회자 정리 ---
    with debate_area:
        st.markdown("---")
        st.markdown("### 🎙️ 사회자 정리 및 개인 의견")
    summary = run_moderator_summary(
        llm=llm,
        topic=topic,
        agent1_name=agent1_name,
        agent1_position=agent1_position,
        agent2_name=agent2_name,
        agent2_position=agent2_position,
        turns=turns,
        display_container=debate_area,
    )
    turns.append({"round": 0, "agent": "사회자", "content": summary})
    return turns


def init_session() -> None:
    """session_state 기본값을 설정한다."""
    if "debate_turns" not in st.session_state:
        st.session_state.debate_turns = []
    if "debate_done" not in st.session_state:
        st.session_state.debate_done = False


def reset_debate() -> None:
    """토론 기록과 진행 상태를 초기화한다."""
    st.session_state.debate_turns = []
    st.session_state.debate_done = False


def render_header() -> None:
    """ref.py 스타일의 헤더(로고 + 타이틀)를 렌더링한다."""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### ⚖️")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">Two Agent</span>
  <span style="color:#ff8c00;">Debate</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def main() -> None:
    st.set_page_config(page_title="Two Agent Debate", page_icon="⚖️", layout="wide")
    init_session()

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

    render_header()

    # ---------------- 사이드바 ----------------
    with st.sidebar:
        model_choice = st.radio(
            "모델 선택",
            ("gpt-4o-mini", "gemini-3-pro-preview", "claude-sonnet-4-5"),
            index=0,
        )
        max_rounds = st.slider("최대 토론횟수", min_value=2, max_value=20, value=5)
        tool_type = st.radio(
            "도구 유형",
            ("인터넷 검색", "문서검색", "도구 없음"),
            index=2,
        )
        st.divider()
        if st.button("다시 시작하기", width="stretch"):
            reset_debate()
            st.rerun()

    # ---------------- 토론 설정 ----------------
    st.subheader("토론 설정")
    topic = st.text_input("토론주제", placeholder="예: 인공지능이 인간의 일자리를 대체해야 하는가?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 에이전트1 설정")
        agent1_name = st.text_input("에이전트 1 이름", value="찬성 측", key="agent1_name")
        agent1_position = st.text_area(
            "에이전트 1 입장설명",
            placeholder="찬성 측의 핵심 주장과 근거를 설명하세요.",
            height=120,
            key="agent1_position",
        )
        agent1_pdfs = None
        if tool_type == "문서검색":
            agent1_pdfs = st.file_uploader(
                "에이전트 1 PDF 문서",
                type=["pdf"],
                accept_multiple_files=True,
                key="agent1_pdf",
            )

    with col2:
        st.markdown("#### 에이전트2 설정")
        agent2_name = st.text_input("에이전트 2 이름", value="반대 측", key="agent2_name")
        agent2_position = st.text_area(
            "에이전트 2 입장설명",
            placeholder="반대 측의 핵심 주장과 근거를 설명하세요.",
            height=120,
            key="agent2_position",
        )
        agent2_pdfs = None
        if tool_type == "문서검색":
            agent2_pdfs = st.file_uploader(
                "에이전트 2 PDF 문서",
                type=["pdf"],
                accept_multiple_files=True,
                key="agent2_pdf",
            )

    start_debate = st.button("토론 시작", type="primary", width="stretch")

    # ---------------- 토론 진행 / 결과 표시 ----------------
    st.divider()
    st.subheader("토론 진행")

    if start_debate:
        reset_debate()

        if not topic.strip():
            st.error("토론주제를 입력해 주세요.")
            st.stop()
        if not agent1_position.strip() or not agent2_position.strip():
            st.error("양쪽 에이전트의 입장설명을 모두 입력해 주세요.")
            st.stop()

        # 문서검색 시 PDF 벡터 스토어 구축
        agent1_vs, agent2_vs = None, None
        if tool_type == "문서검색":
            try:
                with st.spinner("에이전트 1 문서를 처리하는 중..."):
                    if agent1_pdfs:
                        agent1_vs = build_vectorstore(list(agent1_pdfs))
                with st.spinner("에이전트 2 문서를 처리하는 중..."):
                    if agent2_pdfs:
                        agent2_vs = build_vectorstore(list(agent2_pdfs))
            except Exception as exc:  # noqa: BLE001
                st.error(f"문서 처리 중 오류: {exc}")
                st.stop()

        progress_bar = st.progress(0, text="토론 준비 중...")
        debate_area = st.container()

        try:
            turns = run_debate(
                model_name=model_choice,
                max_rounds=max_rounds,
                tool_type=tool_type,
                topic=topic.strip(),
                agent1_name=agent1_name.strip() or "에이전트 1",
                agent1_position=agent1_position.strip(),
                agent2_name=agent2_name.strip() or "에이전트 2",
                agent2_position=agent2_position.strip(),
                agent1_vs=agent1_vs,
                agent2_vs=agent2_vs,
                progress_bar=progress_bar,
                debate_area=debate_area,
            )
            st.session_state.debate_turns = turns
            st.session_state.debate_done = True
        except Exception as exc:  # noqa: BLE001
            st.error(f"토론 중 오류가 발생했습니다: {exc}")

    elif st.session_state.debate_done and st.session_state.debate_turns:
        # 이전 토론 결과 재표시
        for t in st.session_state.debate_turns:
            if t["agent"] == "사회자":
                st.markdown("---")
                st.markdown("### 🎙️ 사회자 정리 및 개인 의견")
            else:
                st.markdown(f"### 🎤 {t['round']}회차 — {t['agent']}")
            st.markdown(remove_separators(t["content"]))
    else:
        st.info("토론 설정을 입력한 뒤 '토론 시작' 버튼을 눌러 주세요.")


if __name__ == "__main__":
    main()
