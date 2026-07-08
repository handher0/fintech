"""
글로벌 핀테크 실시간 FDS 관제 및 리스크 분석 시스템

아키텍처:
  data/global_fds.db (SQLite) → load_data → apply_filters → KPI / 차트 / 리포트

실행:
  cd 5.DatabaseSQL/code
  streamlit run global_fds_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import streamlit as st

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "global_fds.db"
TABLE_NAME = "global_payments"


# ---------------------------------------------------------------------------
# 데이터 로드 & 필터
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="데이터베이스에서 결제 내역을 불러오는 중...")
def load_data(db_path: str) -> pd.DataFrame:
    """SQLite DB에서 global_payments 테이블을 로드하고 파생 컬럼을 추가한다."""
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    except sqlite3.Error as exc:
        raise ConnectionError(f"DB 연결 실패: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"데이터 로드 중 오류: {exc}") from exc

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df


def apply_filters(
    df: pd.DataFrame,
    countries: list[str],
    categories: list[str],
    fraud_filter: str,
    amount_range: tuple[float, float],
) -> pd.DataFrame:
    """사이드바 필터 조건을 DataFrame에 적용한다."""
    filtered = df.copy()

    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if categories:
        filtered = filtered[filtered["merchant_category"].isin(categories)]

    if fraud_filter == "정상 거래만 (0)":
        filtered = filtered[filtered["is_fraud"] == 0]
    elif fraud_filter == "이상거래만 (1)":
        filtered = filtered[filtered["is_fraud"] == 1]

    lo, hi = amount_range
    filtered = filtered[(filtered["amount_usd"] >= lo) & (filtered["amount_usd"] <= hi)]
    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI & 리포트
# ---------------------------------------------------------------------------
def calculate_kpi(df: pd.DataFrame) -> dict:
    """핵심 금융 보안 KPI를 계산한다."""
    total_txn = len(df)
    fraud_df = df[df["is_fraud"] == 1]
    fraud_count = len(fraud_df)
    fraud_ratio = (fraud_count / total_txn * 100) if total_txn else 0.0

    total_amount = df["amount_usd"].sum()
    blocked_amount = fraud_df["amount_usd"].sum()

    # 최고 위험 업종 (이상거래 비율 기준)
    risk_by_cat = (
        df.groupby("merchant_category")
        .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"))
        .assign(ratio=lambda x: x["fraud"] / x["total"] * 100)
    )
    risk_by_country = (
        df.groupby("country")
        .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"))
        .assign(ratio=lambda x: x["fraud"] / x["total"] * 100)
    )

    top_cat = risk_by_cat["ratio"].idxmax() if not risk_by_cat.empty else "-"
    top_cat_ratio = risk_by_cat["ratio"].max() if not risk_by_cat.empty else 0.0
    top_country = risk_by_country["ratio"].idxmax() if not risk_by_country.empty else "-"
    top_country_ratio = risk_by_country["ratio"].max() if not risk_by_country.empty else 0.0

    highest_risk = (
        f"{top_cat} ({top_cat_ratio:.1f}%)"
        if top_cat_ratio >= top_country_ratio
        else f"{top_country} ({top_country_ratio:.1f}%)"
    )

    return {
        "total_txn": total_txn,
        "unique_users": df["user_id"].nunique(),
        "unique_ips": df["device_ip"].nunique(),
        "total_amount": total_amount,
        "fraud_count": fraud_count,
        "fraud_ratio": fraud_ratio,
        "blocked_amount": blocked_amount,
        "highest_risk": highest_risk,
    }


def generate_report(df: pd.DataFrame, kpi: dict) -> str:
    """통계 결과를 조합해 동적 FDS 보안 관제 리포트를 생성한다."""
    if df.empty:
        return "필터 조건에 해당하는 거래 데이터가 없습니다."

    fraud_df = df[df["is_fraud"] == 1]
    period_start = df["timestamp"].min()
    period_end = df["timestamp"].max()
    period_str = ""
    if pd.notna(period_start) and pd.notna(period_end):
        period_str = f"{period_start:%Y년 %m월 %d일} ~ {period_end:%Y년 %m월 %d일}"

    # 새벽 1~4시 고액 이상거래 패턴
    dawn_fraud = fraud_df[(fraud_df["hour"] >= 1) & (fraud_df["hour"] <= 4)]
    dawn_note = ""
    if not dawn_fraud.empty:
        top_dawn_cats = (
            dawn_fraud.groupby("merchant_category")["amount_usd"]
            .sum()
            .sort_values(ascending=False)
            .head(2)
            .index.tolist()
        )
        dawn_note = (
            f"특히 새벽 1시~4시 사이에 '{', '.join(top_dawn_cats)}' 카테고리의 "
            f"고액 결제({len(dawn_fraud)}건)가 주요 이상 패턴으로 분석됩니다."
        )

    # 비-US 지역 이상거래
    non_us_fraud = fraud_df[fraud_df["country"] != "US"]
    region_note = ""
    if len(non_us_fraud) > len(fraud_df) * 0.5:
        top_countries = (
            non_us_fraud["country"].value_counts().head(2).index.tolist()
        )
        region_note = f"비-US 지역({', '.join(top_countries)})에서 이상거래가 집중되고 있습니다."

    # 고액 임계값 (이상거래 75 percentile)
    threshold = fraud_df["amount_usd"].quantile(0.75) if not fraud_df.empty else 0

    lines = [
        f"**{period_str}** 글로벌 결제 관제 결과, "
        f"총 **{kpi['total_txn']:,}건**의 거래 중 "
        f"**{kpi['fraud_count']:,}건**의 이상거래가 탐지되었습니다 "
        f"(발생 비율 **{kpi['fraud_ratio']:.2f}%**).",
        f"이를 통해 총 **${kpi['blocked_amount']:,.2f}** 의 금융 피해를 예방한 것으로 추정됩니다.",
        f"최고 위험 영역은 **{kpi['highest_risk']}** 입니다.",
    ]
    if region_note:
        lines.append(region_note)
    if dawn_note:
        lines.append(dawn_note)
    if threshold > 0:
        lines.append(
            f"이상거래 금액 분포 기준, **${threshold:,.0f}** 이상 고액 결제에 대한 "
            f"FDS 차단 룰 강화를 권고합니다."
        )
    lines.append("해당 Rule의 차단 강도를 높이고, Top 위험 사용자에 대한 계정 동결 검토를 권장합니다.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def _fraud_label(series: pd.Series) -> pd.Series:
    return series.map({0: "정상", 1: "이상거래"})


def draw_daily_trend(df: pd.DataFrame) -> go.Figure:
    """일별 리스크 발생 추세 (Line Chart)."""
    daily = (
        df.groupby(["date", "is_fraud"], as_index=False)
        .agg(amount=("amount_usd", "sum"), count=("transaction_id", "count"))
    )
    daily["risk"] = _fraud_label(daily["is_fraud"])
    fig = px.line(
        daily,
        x="date",
        y="count",
        color="risk",
        markers=True,
        title="일별 이상거래 발생 추세",
        labels={"date": "결제 일자", "count": "거래 건수", "risk": "리스크 등급"},
    )
    fig.update_layout(hovermode="x unified", legend_title="리스크 등급")
    return fig


def draw_country_category_bar(df: pd.DataFrame, dimension: str) -> go.Figure:
    """국가별/업종별 정상 vs 이상거래 금액 비교 (Grouped Bar)."""
    grouped = (
        df.groupby([dimension, "is_fraud"], as_index=False)["amount_usd"]
        .sum()
        .assign(risk=lambda x: _fraud_label(x["is_fraud"]))
    )
    fig = px.bar(
        grouped,
        x=dimension,
        y="amount_usd",
        color="risk",
        barmode="group",
        title=f"{dimension}별 정상/이상거래 금액 비교",
        labels={dimension: dimension, "amount_usd": "총 결제액 (USD)", "risk": "리스크 등급"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def draw_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """정상 vs 이상거래 결제 금액 분포 (Box Plot)."""
    plot_df = df.copy()
    plot_df["risk"] = _fraud_label(plot_df["is_fraud"])
    fig = px.box(
        plot_df,
        x="risk",
        y="amount_usd",
        color="risk",
        title="결제 금액 분포 비교 (정상 vs 이상거래)",
        labels={"amount_usd": "결제 금액 (USD)", "risk": "리스크 등급"},
    )
    return fig


def draw_amount_histogram(df: pd.DataFrame) -> go.Figure:
    """결제 금액 히스토그램."""
    plot_df = df.copy()
    plot_df["risk"] = _fraud_label(plot_df["is_fraud"])
    fig = px.histogram(
        plot_df,
        x="amount_usd",
        color="risk",
        barmode="overlay",
        opacity=0.7,
        nbins=30,
        title="결제 금액 분포 히스토그램",
        labels={"amount_usd": "결제 금액 (USD)", "risk": "리스크 등급"},
    )
    return fig


def draw_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """시간대 × 국가 이상거래 히트맵."""
    fraud = df[df["is_fraud"] == 1]
    if fraud.empty:
        pivot = pd.DataFrame(0, index=sorted(df["country"].unique()), columns=range(24))
    else:
        pivot = fraud.pivot_table(
            index="country",
            columns="hour",
            values="transaction_id",
            aggfunc="count",
            fill_value=0,
        )
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0
        pivot = pivot[sorted(pivot.columns)]

    fig = px.imshow(
        pivot,
        labels=dict(x="결제 시간 (시)", y="국가", color="이상거래 건수"),
        title="시간대 × 국가 리스크 매트릭스 (이상거래 건수)",
        aspect="auto",
        color_continuous_scale="Reds",
    )
    fig.update_xaxes(side="bottom")
    return fig


def draw_top_risk_users(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """위험 사용자 Top-N (Horizontal Bar)."""
    fraud = df[df["is_fraud"] == 1]
    if fraud.empty:
        return go.Figure().update_layout(title=f"위험 사용자 Top {top_n} (데이터 없음)")

    top_users = (
        fraud.groupby("user_id")
        .agg(fraud_count=("transaction_id", "count"), fraud_amount=("amount_usd", "sum"))
        .sort_values("fraud_amount", ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(
        top_users,
        x="fraud_amount",
        y="user_id",
        orientation="h",
        title=f"위험 사용자 Top {top_n} (이상거래 금액 기준)",
        labels={"fraud_amount": "이상거래 금액 (USD)", "user_id": "사용자 ID"},
        text="fraud_count",
    )
    fig.update_traces(texttemplate="%{text}건", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def draw_charts(df: pd.DataFrame) -> None:
    """분석 목적별 Plotly 차트 섹션을 렌더링한다."""
    st.subheader("📊 리스크 분석 시각화")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["일별 추세", "국가/업종 비교", "금액 분포", "리스크 히트맵", "위험 사용자"]
    )

    with tab1:
        st.plotly_chart(draw_daily_trend(df), width="stretch")

    with tab2:
        dim = st.radio("비교 기준", ["country", "merchant_category"], horizontal=True, key="bar_dim")
        label = "국가" if dim == "country" else "업종"
        st.plotly_chart(draw_country_category_bar(df, dim), width="stretch")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(draw_amount_distribution(df), width="stretch")
        with c2:
            st.plotly_chart(draw_amount_histogram(df), width="stretch")

    with tab4:
        st.plotly_chart(draw_risk_heatmap(df), width="stretch")

    with tab5:
        top_n = st.slider("Top N", min_value=5, max_value=10, value=10, step=5, key="top_n")
        st.plotly_chart(draw_top_risk_users(df, top_n), width="stretch")


def highlight_fraud_rows(row: pd.Series) -> list[str]:
    """이상거래 행을 붉은 배경으로 하이라이트한다."""
    color = "background-color: #ffe0e0" if row["is_fraud"] == 1 else ""
    return [color] * len(row)


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], list[str], str, tuple[float, float]]:
    """사이드바 필터 UI를 렌더링하고 선택값을 반환한다."""
    st.sidebar.header("🔍 실시간 필터")

    all_countries = sorted(df["country"].dropna().unique().tolist())
    all_categories = sorted(df["merchant_category"].dropna().unique().tolist())
    amt_min = float(df["amount_usd"].min())
    amt_max = float(df["amount_usd"].max())

    countries = st.sidebar.multiselect(
        "국가 (country)",
        options=all_countries,
        default=all_countries,
    )
    categories = st.sidebar.multiselect(
        "업종 (merchant_category)",
        options=all_categories,
        default=all_categories,
    )
    fraud_filter = st.sidebar.radio(
        "리스크 등급 (is_fraud)",
        ["전체", "정상 거래만 (0)", "이상거래만 (1)"],
        index=0,
    )
    amount_range = st.sidebar.slider(
        "결제 금액 범위 (USD)",
        min_value=amt_min,
        max_value=amt_max,
        value=(amt_min, amt_max),
    )

    st.sidebar.divider()
    st.sidebar.caption(f"DB: `{DB_PATH.name}`")
    st.sidebar.caption(f"전체 레코드: {len(df):,}건")

    return countries, categories, fraud_filter, amount_range


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="글로벌 FDS 관제 대시보드",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ 글로벌 핀테크 실시간 FDS 관제 및 리스크 분석 시스템")
    st.caption("Global Fraud Detection System — 실시간 이상거래 모니터링 · 리스크 분석 · Rule 고도화")

    # DB 연결
    if not DB_PATH.exists():
        st.error(
            f"데이터베이스 파일을 찾을 수 없습니다: `{DB_PATH}`\n\n"
            "먼저 `build_global_fds_db.py`를 실행해 global_fds.db를 생성하세요."
        )
        st.stop()

    try:
        df = load_data(str(DB_PATH))
    except (ConnectionError, RuntimeError) as exc:
        st.error(f"데이터베이스 연결 오류: {exc}")
        st.stop()

    if df.empty:
        st.warning("global_payments 테이블에 데이터가 없습니다.")
        st.stop()

    # 사이드바 필터
    countries, categories, fraud_filter, amount_range = render_sidebar(df)
    filtered = apply_filters(df, countries, categories, fraud_filter, amount_range)

    # 필터 요약
    st.subheader("📋 데이터 탐색 및 리스크 요약")
    kpi = calculate_kpi(filtered)
    s1, s2, s3 = st.columns(3)
    s1.metric("총 거래 건수", f"{kpi['total_txn']:,}")
    s2.metric("고유 사용자 수", f"{kpi['unique_users']:,}")
    s3.metric("탐지된 IP 수", f"{kpi['unique_ips']:,}")

    # KPI 카드
    st.subheader("💰 핵심 금융 보안 KPI")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 결제 모니터링 규모", f"${kpi['total_amount']:,.2f}")
    m2.metric(
        "FDS 이상거래 탐지 건수",
        f"{kpi['fraud_count']:,}건",
        f"{kpi['fraud_ratio']:.2f}%",
    )
    m3.metric("피해 예방 차단 금액", f"${kpi['blocked_amount']:,.2f}")
    m4.metric("최고 위험 업종/국가", kpi["highest_risk"])

    # 데이터 테이블 (이상거래 하이라이트)
    display_cols = [
        "transaction_id", "user_id", "timestamp", "amount_usd",
        "country", "merchant_category", "device_ip", "is_fraud",
    ]
    display_df = filtered[display_cols].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(
        display_df.style.apply(highlight_fraud_rows, axis=1),
        width="stretch",
        height=350,
    )

    # 시각화
    if filtered.empty:
        st.warning("현재 필터 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
    else:
        draw_charts(filtered)

    # AI 기반 동적 리포트
    st.subheader("📝 FDS 보안 관제 자동 요약 리포트")
    report = generate_report(filtered, kpi)
    st.info(report)


if __name__ == "__main__":
    main()
