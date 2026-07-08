"""
거시경제 지표 간 상관관계 분석 및 통계 시각화

한국은행 ECOS API 데이터를 월별로 정렬·결합한 뒤
피어슨/스피어만 상관계수, Plotly 시각화, OpenAI 통계 해석 챗봇을 제공한다.

실행 방법:
  pip install streamlit plotly matplotlib seaborn python-dotenv openai pandas requests numpy
  cd 7.BoKMacro
  streamlit run code/analyze_correlation.py
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from openai import OpenAI
from plotly.subplots import make_subplots

from macro_data import (
    CHARTS_DIR,
    INDICATOR_CONFIGS,
    _end_date_for_cycle,
    fetch_indicator,
    load_merged_env,
    resolve_api_keys,
    show_key_error,
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
ANALYSIS_START_D = "20250101"
ANALYSIS_START_M = "202501"

RECOMMENDED_QUESTIONS = [
    "금리와 환율의 상관관계가 어때?",
    "최근 물가가 오를 때 환율은 어떻게 움직였어?",
    "기준금리, 환율, CPI YoY 중 상관관계가 가장 강한 지표 쌍은 무엇이야?",
]

CORR_SYSTEM_PROMPT = """너는 복잡한 계량경제학 통계 수치를 일반 투자자도 이해하기 쉽게 풀어 설명해 주는 금융 데이터 사이언스 연구원이다.

답변 형식 규칙 (반드시 준수):
- 마크다운 기호(#, ##, **, `, >, ---)를 절대 사용하지 마세요.
- 일반 문단 텍스트와 불렛 리스트(-)만 사용하세요.
- 제공된 피어슨·스피어만 상관계수 수치를 근거로 통계적 사실만 설명하세요.
- 상관관계를 인과관계로 단정하지 마세요.
"""

# 히트맵 분석 대상 (내부 컬럼명 → 한글 라벨)
HEATMAP_COLUMNS = {
    "exchange_avg": "원/달러 환율",
    "base_rate": "기준금리",
    "cpi": "CPI 지수",
    "cpi_yoy": "CPI YoY",
}

# 시각화·선택용 전체 지표
SELECTABLE_COLUMNS = {
    "exchange_avg": "원/달러 환율(월평균)",
    "base_rate": "기준금리",
    "cpi": "CPI 지수",
    "cpi_yoy": "CPI YoY",
    "exchange_mom": "환율 MoM",
    "exchange_yoy": "환율 YoY",
    "rate_mom": "금리 MoM",
    "rate_yoy": "금리 YoY",
    "cpi_mom": "CPI MoM",
}


# ---------------------------------------------------------------------------
# 데이터 결합
# ---------------------------------------------------------------------------
def _daily_to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """일별 환율을 월평균으로 변환한다."""
    if df.empty:
        return pd.DataFrame(columns=["month", "exchange_avg", "DATE"])
    s = df.set_index("DATE")["VALUE"].resample("ME").mean()
    out = s.reset_index()
    out.columns = ["DATE", "exchange_avg"]
    out["month"] = out["DATE"].dt.strftime("%Y-%m")
    return out[["month", "DATE", "exchange_avg"]]


def _to_monthly_series(df: pd.DataFrame, cycle: str, value_col: str) -> pd.DataFrame:
    """월별 시계열로 정규화한다 (일별 금리는 월말 값 사용)."""
    if df.empty:
        return pd.DataFrame(columns=["month", value_col, "DATE"])
    if cycle == "D":
        s = df.set_index("DATE")["VALUE"].resample("ME").last()
    else:
        s = df.set_index("DATE")["VALUE"]
    out = s.reset_index()
    out.columns = ["DATE", value_col]
    out["month"] = out["DATE"].dt.strftime("%Y-%m")
    return out[["month", "DATE", value_col]]


@st.cache_data(ttl=3600, show_spinner="거시경제 데이터 수집 및 결합 중...")
def build_master_dataframe(api_key: str) -> pd.DataFrame:
    """환율·금리·CPI를 월 단위로 inner join한 마스터 DataFrame을 구축한다."""
    end_d = _end_date_for_cycle("D")
    end_m = _end_date_for_cycle("M")

    exchange_df, _ = fetch_indicator(
        api_key, INDICATOR_CONFIGS["exchange"],
        ANALYSIS_START_D, end_d, ANALYSIS_START_M, end_m,
    )
    rate_df, rate_cycle = fetch_indicator(
        api_key, INDICATOR_CONFIGS["rate"],
        ANALYSIS_START_D, end_d, ANALYSIS_START_M, end_m,
    )
    cpi_df, cpi_cycle = fetch_indicator(
        api_key, INDICATOR_CONFIGS["cpi"],
        ANALYSIS_START_D, end_d, ANALYSIS_START_M, end_m,
    )

    ex_monthly = _daily_to_monthly_mean(exchange_df)
    rate_monthly = _to_monthly_series(rate_df, rate_cycle, "base_rate")
    cpi_monthly = _to_monthly_series(cpi_df, cpi_cycle, "cpi")

    master = ex_monthly.merge(
        rate_monthly[["month", "base_rate"]], on="month", how="inner"
    )
    master = master.merge(
        cpi_monthly[["month", "cpi"]], on="month", how="inner"
    )

    master = master.sort_values("DATE").reset_index(drop=True)
    master["cpi_yoy"] = master["cpi"].pct_change(12) * 100
    master["exchange_mom"] = master["exchange_avg"].pct_change(1) * 100
    master["exchange_yoy"] = master["exchange_avg"].pct_change(12) * 100
    master["rate_mom"] = master["base_rate"].pct_change(1) * 100
    master["rate_yoy"] = master["base_rate"].pct_change(12) * 100
    master["cpi_mom"] = master["cpi"].pct_change(1) * 100

    return master


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """피어슨 또는 스피어만 상관계수 행렬을 계산한다 (scipy 미사용)."""
    cols = list(HEATMAP_COLUMNS.keys())
    numeric = df[cols].dropna()
    if numeric.empty or len(numeric) < 2:
        return pd.DataFrame()

    if method == "spearman":
        ranked = numeric.rank()
        corr = ranked.corr(method="pearson")
    else:
        corr = numeric.corr(method="pearson")

    corr.index = [HEATMAP_COLUMNS[c] for c in corr.index]
    corr.columns = [HEATMAP_COLUMNS[c] for c in corr.columns]
    return corr


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """지표별 평균·표준편차 요약표."""
    rows = []
    for col, label in HEATMAP_COLUMNS.items():
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "지표": label,
            "평균": round(s.mean(), 4),
            "표준편차": round(s.std(), 4),
            "최신값": round(s.iloc[-1], 4),
        })
    return pd.DataFrame(rows)


def strongest_pair(corr: pd.DataFrame) -> tuple[str, str, float] | None:
    """자기 자신을 제외한 절대값 최대 상관 쌍을 반환한다."""
    if corr.empty:
        return None
    best_abs = -1.0
    best_pair: tuple[str, str, float] | None = None
    labels = list(corr.columns)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j:
                continue
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) > best_abs:
                best_abs = abs(val)
                best_pair = (a, b, float(val))
    return best_pair


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def build_analysis_context(df: pd.DataFrame, pearson: pd.DataFrame, spearman: pd.DataFrame) -> str:
    parts = [
        f"[분석 기간] {df['month'].iloc[0]} ~ {df['month'].iloc[-1]} (총 {len(df)}개월)",
        "[결합 데이터 최근 6개월]\n" + df.tail(6).to_string(index=False),
        "[피어슨 상관계수]\n" + pearson.round(3).to_string(),
        "[스피어만 상관계수]\n" + spearman.round(3).to_string(),
    ]
    pair = strongest_pair(pearson)
    if pair:
        parts.append(f"[피어슨 최강 상관 쌍] {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
    return "\n\n".join(parts)


def call_llm(client: OpenAI, model: str, question: str, context: str) -> str:
    user_msg = f"[통계 분석 컨텍스트]\n{context}\n\n[사용자 질문]\n{question}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CORR_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return _strip_markdown((resp.choices[0].message.content or "").strip())
    except Exception as exc:
        return f"OpenAI API 호출 중 오류가 발생했습니다: {exc}"


# ---------------------------------------------------------------------------
# Plotly 시각화
# ---------------------------------------------------------------------------
def build_heatmap(corr: pd.DataFrame, method_label: str) -> go.Figure | None:
    if corr.empty:
        return None
    z = corr.values
    labels = list(corr.columns)
    text = [[f"{v:.2f}" if not pd.isna(v) else "" for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text,
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"지표 간 상관계수 히트맵 ({method_label})",
        xaxis_title="",
        yaxis_title="",
        height=480,
    )
    return fig


def build_dual_axis(df: pd.DataFrame, col_a: str, col_b: str) -> go.Figure | None:
    if df.empty or col_a not in df.columns or col_b not in df.columns:
        return None
    plot_df = df.dropna(subset=[col_a, col_b])
    if plot_df.empty:
        return None

    label_a = SELECTABLE_COLUMNS.get(col_a, col_a)
    label_b = SELECTABLE_COLUMNS.get(col_b, col_b)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=plot_df["DATE"], y=plot_df[col_a], name=label_a, mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=plot_df["DATE"], y=plot_df[col_b], name=label_b, mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{label_a} vs {label_b}",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    fig.update_xaxes(tickformat="%Y-%m", dtick="M2", tickangle=-30)
    fig.update_yaxes(title_text=label_a, secondary_y=False)
    fig.update_yaxes(title_text=label_b, secondary_y=True)
    return fig


def build_scatter(df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure | None:
    if df.empty:
        return None
    plot_df = df[[col_x, col_y]].dropna()
    if len(plot_df) < 2:
        return None

    x = plot_df[col_x].values.astype(float)
    y = plot_df[col_y].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    x_line = np.linspace(x.min(), x.max(), 50)
    y_line = slope * x_line + intercept

    label_x = SELECTABLE_COLUMNS.get(col_x, col_x)
    label_y = SELECTABLE_COLUMNS.get(col_y, col_y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="관측치",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        name=f"OLS 추세선 (R={r:.2f})",
    ))
    fig.update_layout(
        title=f"{label_x} vs {label_y}",
        xaxis_title=label_x,
        yaxis_title=label_y,
    )
    return fig


def detect_chat_charts(question: str) -> dict[str, Any]:
    """질문 키워드로 챗봇 하단에 표시할 차트 유형을 추론한다."""
    q = question
    show_heatmap = any(k in q for k in ("상관", "관계", "행렬", "히트맵"))
    show_dual = any(k in q for k in ("비교", "함께", "움직", "추이", "흐름"))
    show_scatter = any(k in q for k in ("산점", "분산", "회귀", "트렌드"))

    col_a, col_b = "base_rate", "exchange_avg"
    if "물가" in q or "cpi" in q.lower():
        col_b = "cpi_yoy" if "yoy" in q.lower() or "전년" in q else "cpi"
    if "금리" in q:
        col_a = "base_rate"
    if "환율" in q or "달러" in q:
        if col_a == "base_rate" and "금리" not in q:
            col_a = "exchange_avg"
        else:
            col_b = "exchange_avg"

    return {
        "heatmap": show_heatmap or ("상관관계" in q),
        "dual": show_dual,
        "scatter": show_scatter or (show_heatmap and not show_dual),
        "col_a": col_a,
        "col_b": col_b,
    }


# ---------------------------------------------------------------------------
# 정적 PNG 백업
# ---------------------------------------------------------------------------
def save_corr_charts_png(
    corr: pd.DataFrame,
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> str | None:
    """히트맵·산점도를 output/charts/corr_{timestamp}.png로 저장한다."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = CHARTS_DIR / f"corr_{ts}.png"

    try:
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if not corr.empty:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                        vmin=-1, vmax=1, ax=axes[0])
            axes[0].set_title("상관계수 히트맵")

        plot_df = df[[col_x := col_a, col_y := col_b]].dropna() if col_a in df.columns else pd.DataFrame()
        if not plot_df.empty and len(plot_df) >= 2:
            x = plot_df[col_x].values.astype(float)
            y = plot_df[col_y].values.astype(float)
            axes[1].scatter(x, y, alpha=0.7)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            r = float(np.corrcoef(x, y)[0, 1])
            axes[1].plot(x_line, slope * x_line + intercept, "r--",
                         label=f"R={r:.2f}")
            axes[1].set_xlabel(SELECTABLE_COLUMNS.get(col_x, col_x))
            axes[1].set_ylabel(SELECTABLE_COLUMNS.get(col_y, col_y))
            axes[1].set_title("산점도 및 OLS 추세선")
            axes[1].legend()
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close("all")
        return str(filepath)
    except Exception:
        plt.close("all")
        return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_summary(df: pd.DataFrame) -> None:
    st.metric("총 분석 개월 수", f"{len(df)}개월")
    if not df.empty:
        st.caption(f"분석 구간: {df['month'].iloc[0]} ~ {df['month'].iloc[-1]}")
    st.dataframe(summary_stats(df), use_container_width=True, hide_index=True)


def render_visualization_tab(df: pd.DataFrame, corr_method: str) -> None:
    pearson = correlation_matrix(df, "pearson")
    spearman = correlation_matrix(df, "spearman")
    active = pearson if corr_method == "pearson" else spearman
    method_label = "피어슨" if corr_method == "pearson" else "스피어만"

    st.subheader("1. 상관계수 히트맵")
    hm = build_heatmap(active, method_label)
    if hm:
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.warning("히트맵을 그릴 데이터가 부족합니다.")

    st.subheader("2. 이중 축 시계열 복합 대조")
    cols = list(SELECTABLE_COLUMNS.keys())
    c1, c2 = st.columns(2)
    with c1:
        sel_a = st.selectbox("좌측 Y축 지표", cols, index=cols.index("base_rate"), key="dual_a")
    with c2:
        sel_b = st.selectbox("우측 Y축 지표", cols, index=cols.index("exchange_avg"), key="dual_b")
    dual = build_dual_axis(df, sel_a, sel_b)
    if dual:
        st.plotly_chart(dual, use_container_width=True)

    st.subheader("3. 산점도 및 OLS 추세선")
    c3, c4 = st.columns(2)
    with c3:
        sel_x = st.selectbox("X축 지표", cols, index=cols.index("base_rate"), key="scatter_x")
    with c4:
        sel_y = st.selectbox("Y축 지표", cols, index=cols.index("exchange_avg"), key="scatter_y")
    sc = build_scatter(df, sel_x, sel_y)
    if sc:
        st.plotly_chart(sc, use_container_width=True)

    if st.button("정적 차트 PNG 백업 저장", use_container_width=True):
        path = save_corr_charts_png(active, df, sel_x, sel_y)
        if path:
            st.success(f"저장 완료: {path}")
        else:
            st.error("PNG 저장에 실패했습니다.")


def process_question(
    question: str,
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    corr_method: str,
) -> None:
    pearson = correlation_matrix(df, "pearson")
    spearman = correlation_matrix(df, "spearman")
    context = build_analysis_context(df, pearson, spearman)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("통계 분석 중..."):
            answer = call_llm(client, model, question, context)
        st.markdown(answer)

        hints = detect_chat_charts(question)
        active = pearson if corr_method == "pearson" else spearman
        method_label = "피어슨" if corr_method == "pearson" else "스피어만"
        charts_shown = False

        if hints["heatmap"] and not active.empty:
            hm = build_heatmap(active, method_label)
            if hm:
                st.plotly_chart(hm, use_container_width=True)
                charts_shown = True

        if hints["dual"]:
            dual = build_dual_axis(df, hints["col_a"], hints["col_b"])
            if dual:
                st.plotly_chart(dual, use_container_width=True)
                charts_shown = True

        if hints["scatter"]:
            sc = build_scatter(df, hints["col_a"], hints["col_b"])
            if sc:
                st.plotly_chart(sc, use_container_width=True)
                charts_shown = True

        if charts_shown:
            save_corr_charts_png(active, df, hints["col_a"], hints["col_b"])

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_chat_tab(df: pd.DataFrame, client: OpenAI, model: str, corr_method: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("**추천 질문**")
    for i, q in enumerate(RECOMMENDED_QUESTIONS):
        if st.button(q, key=f"rec_q_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    prompt = st.session_state.pending_question
    if prompt:
        st.session_state.pending_question = None
        process_question(prompt, df, client, model, corr_method)
        st.rerun()
        return

    if user_input := st.chat_input("금리 인상이 환율과 물가에 미치는 영향에 대해 질문하세요."):
        process_question(user_input, df, client, model, corr_method)
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="거시경제 상관관계 분석",
        page_icon="📊",
        layout="wide",
    )

    merged_env, checked_paths = load_merged_env()
    ecos_key, openai_key, missing = resolve_api_keys(merged_env)

    st.title("📊 거시경제 지표 간 상관관계 분석 및 통계 시각화")
    st.caption("데이터 구간: 2025년 1월 ~ 현재 (환율 월평균 · 금리 · CPI inner join)")

    with st.sidebar:
        st.header("제어 패널")
        model = st.selectbox("LLM 모델", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0)
        corr_method = st.radio(
            "상관계수 알고리즘",
            ["pearson", "spearman"],
            format_func=lambda x: "피어슨 (Pearson)" if x == "pearson" else "스피어만 (Spearman)",
            index=0,
        )
        if st.button("대화 초기화 (Reset Chat)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    if missing:
        show_key_error(missing, checked_paths)
        return

    client = OpenAI(api_key=openai_key)

    try:
        df = build_master_dataframe(api_key=ecos_key)
    except Exception as exc:
        st.error(f"데이터 수집·결합 중 오류가 발생했습니다: {exc}")
        return

    if df.empty or len(df) < 3:
        st.error("결합된 분석 데이터가 부족합니다. ECOS API 응답을 확인해 주세요.")
        return

    render_summary(df)
    st.divider()

    tab_chat, tab_viz = st.tabs(["AI 통계 해석", "상관관계 시각화"])
    with tab_chat:
        render_chat_tab(df, client, model, corr_method)
    with tab_viz:
        render_visualization_tab(df, corr_method)


if __name__ == "__main__":
    main()
