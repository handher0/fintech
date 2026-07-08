"""Streamlit UI 컴포넌트 도감 — Vibe Coding용 샘플러."""

import time

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Streamlit Component Sampler",
    page_icon="📘",
    layout="wide",
)

st.title("Streamlit UI 컴포넌트 도감")
st.caption("Vibe Coding을 위한 화면 용어 및 시각적 예시 정리")

# ------------------------------------------------------------------
# 사이드바 (st.sidebar)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("1. 사이드바 영역 (st.sidebar)")
    st.write(
        "사이드바는 화면 왼쪽에 고정되는 보조 영역입니다. "
        "설정, 필터, 모드 전환 등에 자주 사용합니다."
    )

    mode = st.radio(
        "모드 선택 (st.radio)",
        ["학습 모드", "실전 모드"],
        index=0,
    )

    st.divider()

    st.info(
        f"현재 모드: **{mode}**\n\n"
        "사이드바의 `st.info`는 안내·도움말 메시지를 표시할 때 사용합니다."
    )

# ------------------------------------------------------------------
# 메인 화면 — 4개 탭
# ------------------------------------------------------------------
tab_input, tab_data, tab_layout, tab_status = st.tabs(
    ["입력 도구 (Input)", "데이터 (Data)", "레이아웃 (Layout)", "피드백 (Status)"]
)

# ==================================================================
# 탭 1: 입력 도구 (Input)
# ==================================================================
with tab_input:
    st.subheader("사용자 입력 (Input Widgets)")

    col_left, col_right = st.columns(2)
    memo = ""

    with col_left:
        name = st.text_input(
            "이름 입력 (st.text_input)",
            placeholder="이름을 입력하세요",
        )
        age = st.number_input(
            "나이 입력 (st.number_input)",
            min_value=0,
            max_value=120,
            value=25,
        )
        agreed = st.checkbox("동의 여부 (st.checkbox)", value=False)

    with col_right:
        tags = st.multiselect(
            "관심 태그 선택 (st.multiselect)",
            ["Python", "Streamlit", "AI", "데이터", "웹개발"],
            default=["Python", "Streamlit"],
        )
        difficulty = st.slider(
            "난이도 조절 (st.slider)",
            min_value=1,
            max_value=10,
            value=5,
        )
        with st.expander("팝업 대체 확장기 (st.expander)"):
            st.write("expander 안에 다른 위젯을 넣을 수 있습니다.")
            memo = st.text_input(
                "메모 입력 (st.text_input)",
                placeholder="간단한 메모를 남겨 보세요",
                key="expander_memo",
            )

    st.divider()
    st.write("**입력값 실시간 확인 (st.write, st.json)**")
    st.json(
        {
            "이름": name,
            "나이": age,
            "동의": agreed,
            "관심_태그": tags,
            "난이도": difficulty,
            "메모": memo,
            "모드": mode,
        }
    )

# ==================================================================
# 탭 2: 데이터 (Data)
# ==================================================================
with tab_data:
    st.subheader("데이터 시각화 (Data & Charts)")

    # 랜덤 데이터 10행 3열
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "온도": np.random.randint(15, 35, size=10),
            "습도": np.random.randint(30, 90, size=10),
            "풍속": np.random.randint(1, 20, size=10),
        }
    )

    col_data_left, col_data_right = st.columns([1, 2])

    with col_data_left:
        st.write("데이터프레임 표시 (st.dataframe)")
        st.dataframe(df, width="stretch")

        st.write("정적 테이블 (st.table) — 상위 3행")
        st.table(df.head(3))

    with col_data_right:
        st.write("라인 차트 (st.line_chart)")
        st.line_chart(df)

        st.write("지표 표시 (st.metric)")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("온도", f"{df['온도'].iloc[-1]}°C", f"{df['온도'].iloc[-1] - df['온도'].iloc[-2]:+d}°C")
        with metric_col2:
            st.metric("습도", f"{df['습도'].iloc[-1]}%", f"{df['습도'].iloc[-1] - df['습도'].iloc[-2]:+d}%")
        with metric_col3:
            st.metric("풍속", f"{df['풍속'].iloc[-1]} m/s", f"{df['풍속'].iloc[-1] - df['풍속'].iloc[-2]:+d} m/s")

# ==================================================================
# 탭 3: 레이아웃 (Layout)
# ==================================================================
with tab_layout:
    st.subheader("구조 잡기 (Layouts)")

    with st.expander("눌러서 내용 보기 (st.expander)"):
        st.write(
            "expander는 접었다 펼 수 있는 영역입니다. "
            "부가 설명이나 숨겨진 옵션을 배치할 때 유용합니다."
        )
        st.image(
            "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
            caption="Streamlit 로고 (st.image)",
            width=300,
        )

    with st.container(border=True):
        st.write(
            "컨테이너 (st.container, border=True)는 관련 요소를 "
            "하나의 박스로 묶어 시각적으로 구분합니다."
        )
        if st.button("버튼 (st.button)", key="layout_button"):
            st.write("버튼이 클릭되었습니다!")

# ==================================================================
# 탭 4: 피드백 (Status)
# ==================================================================
with tab_status:
    st.subheader("알림 및 상태 (Status)")

    if st.button("성공 메시지 표시 (st.button)", key="success_btn"):
        st.success("작업이 성공적으로 완료되었습니다! (st.success)")
        st.toast("토스트 알림이 표시되었습니다! (st.toast)")

    if st.button("로딩 효과 표시 (st.button)", key="loading_btn"):
        with st.spinner("처리 중입니다... (st.spinner)"):
            time.sleep(2)
        st.success("2초 후 처리가 완료되었습니다!")

    st.warning("주의가 필요한 상황을 알릴 때 사용합니다. (st.warning)")
    st.error("오류가 발생했을 때 사용합니다. (st.error)")
