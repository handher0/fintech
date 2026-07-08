"""4주차. one-agent.py

AI Agent가 사용자의 질문을 분석해 4개의 tool(시간/챗봇/인터넷검색/RAG) 중
가장 적합한 1개를 자동으로 선택해 실행하는 통합 앱.

- 기존 one.py의 수동 메뉴 선택을 제거하고 tool 선택을 agent에 위임한다.
- callback handler로 agent가 선택한 tool 이름을 기억해 사이드바에 표시한다.
- 각 tool은 3.first/code의 time.py / chatbot.py / internet.py / rag.py 동작을 수행한다.
"""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ------------------------------------------------------------------
# 공통: API 키 로드
# ------------------------------------------------------------------
def load_api_key() -> str | None:
    """실행 위치와 무관하게, 이 파일 기준으로 저장소 루트의 .env를 로드한다."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break
    else:
        load_dotenv(override=False)
    key = os.getenv("OPENAI_API_KEY")
    return key.strip() if key else None


API_KEY = load_api_key()

# tool의 실제 함수명 -> 사이드바에 표시할 한글 이름
TOOL_DISPLAY_NAMES = {
    "get_current_time": "시간",
    "general_chatbot": "챗봇",
    "internet_search": "인터넷검색",
    "rag_search": "RAG",
}

CHATBOT_SYSTEM_PROMPT = "당신은 숭실대학교의 친절한 AI 도우미입니다. 한국어로 답변하세요."

RAG_SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. "
    "말투는 존대말 한글로 해줘."
)

AGENT_SYSTEM_PROMPT = """당신은 사용자의 질문을 분석해서 아래 4개의 tool 중 가장 적합한 1개를 반드시 선택해 실행하는 AI Agent입니다.

- get_current_time (시간): 현재 시간, 날짜, 요일 등을 묻는 질문
- general_chatbot (챗봇): 일반 대화, 상식·지식 질문 등 다른 tool이 필요 없는 질문
- internet_search (인터넷검색): 최신 뉴스, 실시간 정보, 검색이 필요한 질문
- rag_search (RAG): 업로드된 PDF 문서 내용에 대한 질문

반드시 위 tool 중 정확히 1개를 호출하세요. tool 없이 직접 답변하지 마세요."""


# ------------------------------------------------------------------
# Callback Handler: agent가 선택한 tool 이름을 기억한다
# ------------------------------------------------------------------
class ToolSelectionCallbackHandler(BaseCallbackHandler):
    """tool 실행이 시작될 때 tool 이름을 session_state에 기록하는 handler."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "")
        st.session_state.selected_tool = TOOL_DISPLAY_NAMES.get(name, name)


# ==================================================================
# Tool 1) 시간 (time.py)
# ==================================================================
def render_clock():
    """time.py의 실시간 디지털 시계를 렌더링한다."""
    components.html(
        r"""
        <!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
        <style>
            html,body{margin:0;padding:0;background:#000;}
            .wrap{width:100%;text-align:center;padding-top:24px;}
            .time{font-family:'Orbitron',monospace;font-size:clamp(48px,12vw,120px);font-weight:700;
                color:#00ff41;text-shadow:0 0 12px rgba(0,255,65,.7),0 0 30px rgba(0,255,65,.4);letter-spacing:4px;}
            .date{font-family:'Share Tech Mono',monospace;font-size:clamp(20px,4vw,40px);color:#ffff00;margin-top:12px;}
        </style></head><body>
            <div class="wrap"><div class="time" id="time">--:--:--</div>
            <div class="date" id="date">----년 --월 --일</div></div>
            <script>
                const days=['일','월','화','수','목','금','토'];
                const p=n=>String(n).padStart(2,'0');
                function tick(){const d=new Date();
                    document.getElementById('time').textContent=p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
                    document.getElementById('date').textContent=d.getFullYear()+'년 '+(d.getMonth()+1)+'월 '+d.getDate()+'일 ('+days[d.getDay()]+'요일)';
                    setTimeout(tick,250);}
                tick();
            </script>
        </body></html>
        """,
        height=260,
    )


@tool(return_direct=True)
def get_current_time(query: str) -> str:
    """현재 시간, 날짜, 요일을 알려준다. '지금 몇 시야', '오늘 며칠이야' 같은 시간/날짜 질문에 사용한다."""
    now = datetime.now()
    days = ["월", "화", "수", "목", "금", "토", "일"]
    return (
        f"지금은 {now.year}년 {now.month}월 {now.day}일 "
        f"({days[now.weekday()]}요일) "
        f"{now.hour:02d}시 {now.minute:02d}분 {now.second:02d}초입니다."
    )


# ==================================================================
# Tool 2) 챗봇 (chatbot.py)
# ==================================================================
def _history_messages() -> list:
    """지금까지의 대화 기록을 LangChain 메시지 리스트로 변환한다."""
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


@tool(return_direct=True)
def general_chatbot(question: str) -> str:
    """일반적인 대화, 상식·지식 질문에 답한다. 시간/인터넷검색/문서(RAG)가 필요 없는 질문에 사용한다."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=API_KEY)
    llm_messages = [SystemMessage(content=CHATBOT_SYSTEM_PROMPT)]
    llm_messages.extend(_history_messages()[:-1])  # 마지막 사용자 입력 제외한 과거 대화
    llm_messages.append(HumanMessage(content=question))
    return llm.invoke(llm_messages).content


# ==================================================================
# Tool 3) 인터넷 검색 (internet.py)
# ==================================================================
def _extract_answer(response):
    """Responses API 응답에서 본문 텍스트와 url_citation(출처)을 추출한다."""
    text_parts, citations = [], []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "output_text":
                if getattr(block, "text", None):
                    text_parts.append(block.text)
                for ann in getattr(block, "annotations", []) or []:
                    if getattr(ann, "type", None) == "url_citation":
                        url = getattr(ann, "url", "")
                        if url:
                            citations.append((getattr(ann, "title", "") or url, url))
    text = "\n".join(text_parts).strip() or (getattr(response, "output_text", "") or "").strip()
    seen, uniq = set(), []
    for t, u in citations:
        if u not in seen:
            seen.add(u)
            uniq.append((t, u))
    return text, uniq


@tool(return_direct=True)
def internet_search(query: str) -> str:
    """최신 뉴스, 실시간 정보, 시사 등 인터넷 검색이 필요한 질문에 답한다."""
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)
    kwargs = {
        "model": "gpt-5-nano",
        "input": query,
        "tools": [{"type": "web_search"}],
    }
    # 멀티턴: 직전 response.id 를 넘겨 대화 이어가기
    if st.session_state.get("internet_prev_id"):
        kwargs["previous_response_id"] = st.session_state.internet_prev_id

    response = client.responses.create(**kwargs)
    st.session_state.internet_prev_id = response.id

    text, citations = _extract_answer(response)
    text = text or "(응답을 생성하지 못했습니다.)"
    if citations:
        text += "\n\n**출처**\n"
        text += "\n".join(f"{i}. [{t}]({u})" for i, (t, u) in enumerate(citations, 1))
    return text


# ==================================================================
# Tool 4) RAG (rag.py)
# ==================================================================
def _extract_pdf_text(uploaded_file) -> str:
    """pypdf로 먼저 추출하고, 결과가 부실하면 pdfplumber로 재시도한다."""
    from pypdf import PdfReader

    text = ""
    try:
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""
    if len(text.strip()) < 20:
        try:
            import pdfplumber

            uploaded_file.seek(0)
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            pass
    return text


def _build_retriever(files):
    """업로드된 PDF들에서 텍스트를 추출·청크 분할하고 앙상블 리트리버를 만든다."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.retrievers import EnsembleRetriever

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs, empty = [], []
    for f in files:
        text = _extract_pdf_text(f)
        if len(text.strip()) < 20:
            empty.append(f.name)
            continue
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": f.name}))

    if empty:
        st.warning(
            "다음 PDF에서 텍스트를 추출하지 못했습니다(스캔·이미지 PDF 가능성): "
            + ", ".join(empty)
        )
    if not docs:
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.5, 0.5])


@tool(return_direct=True)
def rag_search(question: str) -> str:
    """업로드된 PDF 문서의 내용에 대한 질문에 답한다. 문서·자료·파일 내용을 묻는 질문에 사용한다."""
    retriever = st.session_state.get("rag_retriever")
    if retriever is None:
        return "먼저 사이드바에서 PDF를 업로드하고 '문서 처리'를 눌러 주세요."

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    llm_messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
    llm_messages.extend(_history_messages()[:-1])
    llm_messages.append(
        HumanMessage(
            content=(
                f"다음 문서 내용을 참고해서 질문에 답해 주세요.\n\n"
                f"[문서 내용]\n{context}\n\n[질문]\n{question}"
            )
        )
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=API_KEY)
    return llm.invoke(llm_messages).content


# ==================================================================
# AI Agent 구성
# ==================================================================
TOOLS = [get_current_time, general_chatbot, internet_search, rag_search]


@st.cache_resource
def get_agent_executor() -> AgentExecutor:
    """tool-calling agent와 executor를 생성한다."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=API_KEY)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=False)


# ==================================================================
# 상태 초기화 & 새로 시작
# ==================================================================
def init_session():
    if "messages" not in st.session_state:
        # {"role": "user"|"assistant", "content": str, "tool": str|None}
        st.session_state.messages = []
    if "selected_tool" not in st.session_state:
        st.session_state.selected_tool = None
    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None
    if "internet_prev_id" not in st.session_state:
        st.session_state.internet_prev_id = None


def reset_conversation():
    """지금까지의 대화를 모두 지우고 처음부터 다시 시작한다."""
    st.session_state.messages = []
    st.session_state.selected_tool = None
    st.session_state.internet_prev_id = None


# ==================================================================
# 메인 앱
# ==================================================================
def main():
    st.set_page_config(page_title="AI Agent 통합 앱", page_icon="🤖", layout="wide")
    init_session()

    # ---------------- 사이드바 ----------------
    with st.sidebar:
        st.header("🤖 AI Agent 통합 앱")
        st.caption("질문을 입력하면 AI Agent가 알맞은 tool을 자동으로 선택합니다.")

        st.subheader("🔧 선택된 Tool")
        if st.session_state.selected_tool:
            st.success(f"**{st.session_state.selected_tool}**")
        else:
            st.info("아직 선택된 tool이 없습니다.")

        st.divider()
        st.subheader("📚 RAG 문서")
        uploaded = st.file_uploader(
            "PDF 업로드", type="pdf", accept_multiple_files=True, key="rag_uploader"
        )
        if st.button("문서 처리", use_container_width=True):
            if not uploaded:
                st.warning("먼저 PDF 파일을 업로드해 주세요.")
            else:
                with st.spinner("문서를 처리하는 중..."):
                    retriever = _build_retriever(uploaded)
                if retriever is None:
                    st.error("PDF에서 텍스트를 추출하지 못했습니다.")
                else:
                    st.session_state.rag_retriever = retriever
                    st.success("저장이 끝났습니다! 이제 질문할 수 있습니다.")

        st.divider()
        if st.button("🔄 새로 시작하기", use_container_width=True):
            reset_conversation()
            st.rerun()

    # ---------------- 본문 ----------------
    st.title("🤖 AI Agent 통합 앱")

    if not API_KEY:
        st.error(
            "OPENAI_API_KEY를 찾을 수 없습니다. 저장소 루트의 `.env`에 "
            "`OPENAI_API_KEY=sk-...` 를 추가한 뒤 다시 실행해 주세요."
        )
        st.stop()

    # 지난 대화 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("tool"):
                    st.caption(f"🔧 사용된 tool: {msg['tool']}")
                if msg.get("tool") == "시간":
                    render_clock()

    # 사용자 입력 처리
    if prompt := st.chat_input("무엇이든 물어보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "tool": None})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI Agent가 tool을 선택하는 중..."):
                try:
                    handler = ToolSelectionCallbackHandler()
                    executor = get_agent_executor()
                    result = executor.invoke(
                        {
                            "input": prompt,
                            "chat_history": _history_messages()[:-1],
                        },
                        config={"callbacks": [handler]},
                    )
                    answer = result.get("output", "") or "(응답을 생성하지 못했습니다.)"
                except Exception as e:  # noqa: BLE001
                    answer = f"오류가 발생했습니다: {e}"

            st.markdown(answer)
            used_tool = st.session_state.selected_tool
            if used_tool:
                st.caption(f"🔧 사용된 tool: {used_tool}")
            if used_tool == "시간":
                render_clock()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "tool": used_tool}
        )
        st.rerun()  # 사이드바의 '선택된 Tool' 표시를 즉시 갱신


if __name__ == "__main__":
    main()
