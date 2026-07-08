import streamlit as st

import one

st.set_page_config(page_title="숭실대학교 AI 앱", page_icon="🎓", layout="wide")


def render_home():
    st.title("🎓 숭실대학교 AI 앱")
    st.caption("여러 AI 기능을 한 곳에서 사용할 수 있는 통합 앱입니다.")

    st.markdown(
        """
        왼쪽 사이드바에서 원하는 기능을 선택하세요.

        - **🕒 시간** — 현재 날짜와 시간을 실시간으로 표시
        - **💬 챗봇** — 이전 대화를 기억하는 대화형 챗봇
        - **🌐 인터넷 검색** — 웹 검색을 활용해 답변하는 챗봇
        - **📚 RAG** — 업로드한 PDF 문서 기반 질의응답

        대화를 처음부터 다시 시작하려면 사이드바의 **🔄 새로 시작하기** 버튼을 누르세요.
        """
    )


MENU = {
    "🏠 홈": render_home,
    "🕒 시간": one.render_time,
    "💬 챗봇": one.render_chatbot,
    "🌐 인터넷 검색": one.render_internet,
    "📚 RAG": one.render_rag,
}

with st.sidebar:
    st.header("🎓 메뉴")
    choice = st.radio("이동할 페이지", list(MENU.keys()))
    if st.button("🔄 새로 시작하기", use_container_width=True):
        one.reset_conversations()
        st.rerun()

MENU[choice]()
