import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ------------------------------------------------------------------
# 공통: API 키 로드
# ------------------------------------------------------------------
def load_api_key() -> str | None:
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


def require_api_key() -> bool:
    if not API_KEY:
        st.error(
            "OPENAI_API_KEY를 찾을 수 없습니다. 저장소 루트의 `.env`에 "
            "`OPENAI_API_KEY=sk-...` 를 추가한 뒤 다시 실행해 주세요."
        )
        return False
    return True


# ==================================================================
# 1) 시간
# ==================================================================
def render_time():
    st.title("🕒 시간")
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


# ==================================================================
# 2) 챗봇
# ==================================================================
def render_chatbot():
    st.title("💬 챗봇")
    if not require_api_key():
        return

    from langchain_openai import ChatOpenAI

    system_prompt = "당신은 숭실대학교의 친절한 AI 도우미입니다. 한국어로 답변하세요."
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = [SystemMessage(content=system_prompt)]

    for msg in st.session_state.chatbot_messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.chatbot_messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=API_KEY)
                answer = llm.invoke(st.session_state.chatbot_messages).content
                st.write(answer)
        st.session_state.chatbot_messages.append(AIMessage(content=answer))


# ==================================================================
# 3) 인터넷 검색
# ==================================================================
def _extract_answer(response):
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


def render_internet():
    st.title("🌐 인터넷 검색")
    if not require_api_key():
        return

    from openai import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )

    if "internet_messages" not in st.session_state:
        st.session_state.internet_messages = []
    if "internet_prev_id" not in st.session_state:
        st.session_state.internet_prev_id = None

    def render_ai(content):
        if isinstance(content, str):
            st.markdown(content)
            return
        for block in content:
            if block.get("type") == "text":
                st.markdown(block.get("text", ""))
            elif block.get("type") == "citations":
                st.markdown("**출처**")
                for i, (t, u) in enumerate(block.get("items", []), 1):
                    st.markdown(f"{i}. [{t}]({u})")

    for msg in st.session_state.internet_messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                render_ai(msg.content)

    if prompt := st.chat_input("무엇이든 물어보세요 (인터넷 검색 지원)..."):
        st.session_state.internet_messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("인터넷을 검색하는 중..."):
                try:
                    client = OpenAI(api_key=API_KEY)
                    kwargs = {
                        "model": "gpt-5-nano",
                        "input": prompt,
                        "tools": [{"type": "web_search"}],
                    }
                    if st.session_state.internet_prev_id:
                        kwargs["previous_response_id"] = st.session_state.internet_prev_id
                    response = client.responses.create(**kwargs)
                    st.session_state.internet_prev_id = response.id
                    text, citations = _extract_answer(response)
                    text = text or "(응답을 생성하지 못했습니다.)"
                    st.markdown(text)
                    if citations:
                        st.markdown("**출처**")
                        for i, (t, u) in enumerate(citations, 1):
                            st.markdown(f"{i}. [{t}]({u})")
                    blocks = [{"type": "text", "text": text}]
                    if citations:
                        blocks.append({"type": "citations", "items": citations})
                    st.session_state.internet_messages.append(AIMessage(content=blocks))
                except AuthenticationError:
                    st.error("인증 오류(401): API 키가 유효하지 않습니다. `.env`를 확인해 주세요.")
                except RateLimitError:
                    st.error("요청 한도/할당량 초과입니다. 잠시 후 다시 시도해 주세요.")
                except APIConnectionError:
                    st.error("네트워크 연결 오류입니다. 인터넷 연결을 확인해 주세요.")
                except APIError as e:
                    st.error(f"OpenAI API 오류가 발생했습니다: {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"알 수 없는 오류가 발생했습니다: {e}")


# ==================================================================
# 4) RAG
# ==================================================================
RAG_SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. "
    "말투는 존대말 한글로 해줘."
)


def _extract_pdf_text(uploaded_file) -> str:
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


def render_rag():
    st.title("📚 RAG")
    if not require_api_key():
        return

    from langchain_openai import ChatOpenAI

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None

    with st.sidebar:
        st.divider()
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

    for msg in st.session_state.rag_messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    if question := st.chat_input("문서에 대해 질문해 보세요..."):
        if st.session_state.rag_retriever is None:
            st.chat_message("user").write(question)
            st.chat_message("assistant").write(
                "먼저 사이드바에서 PDF를 업로드하고 '문서 처리'를 눌러 주세요."
            )
        else:
            st.session_state.rag_messages.append(HumanMessage(content=question))
            st.chat_message("user").write(question)
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    docs = st.session_state.rag_retriever.invoke(question)
                    context = "\n\n".join(d.page_content for d in docs)
                    llm_messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
                    llm_messages.extend(st.session_state.rag_messages[:-1])
                    llm_messages.append(
                        HumanMessage(
                            content=(
                                f"다음 문서 내용을 참고해서 질문에 답해 주세요.\n\n"
                                f"[문서 내용]\n{context}\n\n[질문]\n{question}"
                            )
                        )
                    )
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=API_KEY)
                    answer = llm.invoke(llm_messages).content
                    st.write(answer)
            st.session_state.rag_messages.append(AIMessage(content=answer))


# ==================================================================
# 사이드바 & 라우팅
# ==================================================================
MENU = {
    "시간": render_time,
    "챗봇": render_chatbot,
    "인터넷 검색": render_internet,
    "RAG": render_rag,
}

CONVO_KEYS = [
    "chatbot_messages",
    "internet_messages",
    "internet_prev_id",
    "rag_messages",
    "rag_retriever",
]

def reset_conversations():
    for key in CONVO_KEYS:
        st.session_state.pop(key, None)


def run():
    """사이드바 메뉴 + 새로시작 버튼을 렌더링하고 선택된 앱을 실행한다."""
    with st.sidebar:
        st.header("🧩 통합 앱")
        choice = st.radio("메뉴 선택", list(MENU.keys()))
        if st.button("🔄 새로 시작하기", use_container_width=True):
            reset_conversations()
            st.rerun()
    MENU[choice]()


if __name__ == "__main__":
    st.set_page_config(page_title="통합 앱", page_icon="🧩", layout="wide")
    run()
