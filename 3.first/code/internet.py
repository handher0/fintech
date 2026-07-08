import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage


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


st.set_page_config(page_title="🌐 Internet Search Chatbot", page_icon="🌐")
st.title("🌐 Internet Search Chatbot")

api_key = load_api_key()

# 키가 없으면 예외로 종료하지 않고 안내 후 입력 차단
if not api_key:
    st.error(
        "OPENAI_API_KEY를 찾을 수 없습니다. 저장소 루트의 `.env` 파일에 "
        "`OPENAI_API_KEY=sk-...` 형태로 키를 추가한 뒤 다시 실행해 주세요."
    )
    st.stop()

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


@st.cache_resource
def get_client(key: str) -> OpenAI:
    return OpenAI(api_key=key)


MODEL = "gpt-5-nano"


def extract_answer(response) -> tuple[str, list[tuple[str, str]]]:
    """Responses API 응답에서 본문 텍스트와 url_citation(출처)을 추출한다."""
    text_parts: list[str] = []
    citations: list[tuple[str, str]] = []

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "output_text":
                if getattr(block, "text", None):
                    text_parts.append(block.text)
                for ann in getattr(block, "annotations", []) or []:
                    if getattr(ann, "type", None) == "url_citation":
                        title = getattr(ann, "title", "") or getattr(ann, "url", "")
                        url = getattr(ann, "url", "")
                        if url:
                            citations.append((title, url))

    text = "\n".join(text_parts).strip()
    if not text:
        text = (getattr(response, "output_text", "") or "").strip()

    # 중복 출처 제거(순서 유지)
    seen = set()
    unique_citations = []
    for title, url in citations:
        if url not in seen:
            seen.add(url)
            unique_citations.append((title, url))

    return text, unique_citations


def render_ai_content(content) -> None:
    """AIMessage content(문자열 또는 블록 리스트)를 화면에 읽기 좋게 렌더링."""
    if isinstance(content, str):
        st.markdown(content)
        return

    # 블록 리스트 형태 처리
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                st.markdown(block.get("text", ""))
            elif block.get("type") == "citations":
                cites = block.get("items", [])
                if cites:
                    st.markdown("**출처**")
                    for i, (title, url) in enumerate(cites, 1):
                        st.markdown(f"{i}. [{title}]({url})")
        else:
            st.markdown(str(block))


# 대화 기록 & 멀티턴을 위한 상태
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[HumanMessage | AIMessage]
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# 지난 대화 렌더링
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            render_ai_content(msg.content)

# 사용자 입력
if prompt := st.chat_input("무엇이든 물어보세요 (인터넷 검색 지원)..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("인터넷을 검색하는 중..."):
            try:
                client = get_client(api_key)
                kwargs = {
                    "model": MODEL,
                    "input": prompt,
                    "tools": [{"type": "web_search"}],
                }
                # 멀티턴: 직전 response.id 를 넘겨 대화 이어가기
                if st.session_state.previous_response_id:
                    kwargs["previous_response_id"] = st.session_state.previous_response_id

                response = client.responses.create(**kwargs)
                st.session_state.previous_response_id = response.id

                text, citations = extract_answer(response)
                if not text:
                    text = "(응답을 생성하지 못했습니다.)"

                st.markdown(text)
                if citations:
                    st.markdown("**출처**")
                    for i, (title, url) in enumerate(citations, 1):
                        st.markdown(f"{i}. [{title}]({url})")

                # AIMessage 로 상태 저장 (텍스트 + 출처 블록)
                ai_blocks = [{"type": "text", "text": text}]
                if citations:
                    ai_blocks.append({"type": "citations", "items": citations})
                st.session_state.messages.append(AIMessage(content=ai_blocks))

            except AuthenticationError:
                st.error(
                    "인증 오류(401): API 키가 유효하지 않습니다. `.env`의 "
                    "`OPENAI_API_KEY`를 확인해 주세요."
                )
            except RateLimitError:
                st.error(
                    "요청 한도/할당량 초과입니다. 잠시 후 다시 시도하거나 "
                    "OpenAI 사용량(billing)을 확인해 주세요."
                )
            except APIConnectionError:
                st.error("네트워크 연결 오류입니다. 인터넷 연결 상태를 확인해 주세요.")
            except APIError as e:
                st.error(f"OpenAI API 오류가 발생했습니다: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"알 수 없는 오류가 발생했습니다: {e}")
