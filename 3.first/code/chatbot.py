import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
f

def load_api_key() -> str | None:
    """실행 위치와 무관하게, 이 파일 기준 절대경로로 루트의 .env를 로드한다.

    현재 파일 위치: <repo_root>/3.first/code/chatbot.py 이므로
    부모 디렉터리들을 올라가며 .env 를 찾아 명시적으로 로드한다.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break
    else:
        # 마지막 안전장치: 기본 탐색
        load_dotenv(override=False)

    return os.getenv("OPENAI_API_KEY")


st.set_page_config(page_title="숭실대학교 Chatbot", page_icon="🎓")
st.title("숭실대학교 Chatbot")

api_key = load_api_key()

# 환경변수가 없으면 예외로 종료하지 않고 안내 메시지를 보여준다.
if not api_key:
    st.error(
        "OPENAI_API_KEY를 찾을 수 없습니다. 루트 폴더의 `.env` 파일에 "
        "`OPENAI_API_KEY=sk-...` 형태로 키를 추가한 뒤 다시 실행해 주세요."
    )
    st.stop()

# LangChain 0.3: langchain.schema 대신 langchain_core.messages 사용
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


SYSTEM_PROMPT = "당신은 숭실대학교의 친절한 AI 도우미입니다. 한국어로 답변하세요."

# 대화 기록(이전 대화 기억) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# 지난 대화 렌더링 (SystemMessage 제외)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            llm = get_llm()
            # 전체 대화 기록을 함께 전달하여 이전 맥락을 기억
            response = llm.invoke(st.session_state.messages)
            answer = response.content
            st.write(answer)

    st.session_state.messages.append(AIMessage(content=answer))
