import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def load_api_key() -> str | None:
    """실행 위치와 무관하게, 이 파일 기준으로 저장소 루트의 .env를 절대경로로 로드한다."""
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


st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")

api_key = load_api_key()

# 키가 없으면 예외로 종료하지 않고 안내 메시지 표시
if not api_key:
    st.error(
        "OPENAI_API_KEY를 찾을 수 없습니다. 저장소 루트의 `.env` 파일에 "
        "`OPENAI_API_KEY=sk-...` 형태로 키를 추가한 뒤 다시 실행해 주세요."
    )
    st.stop()

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from pypdf import PdfReader

SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. "
    "말투는 존대말 한글로 해줘."
)


def extract_text_from_pdf(uploaded_file) -> str:
    """pypdf로 먼저 추출하고, 결과가 부실하면 pdfplumber로 재시도한다.

    스캔(이미지) PDF는 텍스트 레이어가 없어 두 방법 모두 빈 결과가 나올 수 있다.
    """
    # 1차: pypdf
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""

    # 2차: pypdf 결과가 사실상 비었으면 pdfplumber로 재시도
    if len(text.strip()) < 20:
        try:
            import pdfplumber

            uploaded_file.seek(0)
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            pass

    return text


def build_retriever(files):
    """업로드된 여러 PDF에서 텍스트를 추출·청크 분할하고 앙상블 리트리버를 만든다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs: list[Document] = []
    empty_files: list[str] = []
    for f in files:
        text = extract_text_from_pdf(f)
        if len(text.strip()) < 20:
            # 텍스트 레이어가 없는(스캔/이미지) PDF일 가능성이 큼
            empty_files.append(f.name)
            continue
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": f.name}))

    if empty_files:
        st.warning(
            "다음 PDF에서는 텍스트를 추출하지 못했습니다(스캔·이미지 PDF이거나 "
            "폰트 인코딩 문제일 수 있습니다): "
            + ", ".join(empty_files)
        )

    if not docs:
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)


# ------------------------------------------------------------------
# 상태 초기화
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[HumanMessage | AIMessage]
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------------------------------------------------------
# 사이드바: PDF 업로드 & 처리
# ------------------------------------------------------------------
with st.sidebar:
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드", type="pdf", accept_multiple_files=True
    )
    if st.button("문서 처리", use_container_width=True):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드해 주세요.")
        else:
            with st.spinner("문서를 처리하는 중..."):
                retriever = build_retriever(uploaded_files)
            if retriever is None:
                st.error("PDF에서 텍스트를 추출하지 못했습니다.")
            else:
                st.session_state.retriever = retriever
                st.success("저장이 끝났습니다! 이제 질문할 수 있습니다.")

# ------------------------------------------------------------------
# 지난 대화 렌더링
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ------------------------------------------------------------------
# 사용자 질문 (화면 하단 고정 입력창)
# ------------------------------------------------------------------
if question := st.chat_input("문서에 대해 질문해 보세요..."):
    if st.session_state.retriever is None:
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(
            "먼저 사이드바에서 PDF를 업로드하고 '문서 처리'를 눌러 주세요."
        )
    else:
        st.session_state.messages.append(HumanMessage(content=question))
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                # 관련 정보 검색 (앙상블 리트리버)
                docs = st.session_state.retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)

                # 이전 대화를 기억하도록 전체 히스토리를 전달
                llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                llm_messages.extend(st.session_state.messages[:-1])  # 과거 대화
                llm_messages.append(
                    HumanMessage(
                        content=(
                            f"다음 문서 내용을 참고해서 질문에 답해 주세요.\n\n"
                            f"[문서 내용]\n{context}\n\n[질문]\n{question}"
                        )
                    )
                )

                answer = get_llm().invoke(llm_messages).content
                st.write(answer)

        st.session_state.messages.append(AIMessage(content=answer))
