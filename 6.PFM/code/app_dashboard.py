"""
개인 맞춤형 자산 관리 가계부 분석 대시보드

구동 환경:
  Python 3.10+, streamlit, pandas, plotly
  데이터: ./6.PFM/data/user_card_history_merged.csv
    컬럼: month, Approved_Num, Approved_DateTime, Card_Name, Store_Name, Category, Amount_KRW

실행 방법:
  cd 6.PFM/code
  streamlit run app_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "user_card_history_merged.csv"

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]


# ---------------------------------------------------------------------------
# 데이터 로드 & 전처리
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="가계부 데이터를 불러오는 중...")
def load_data(csv_path: str) -> pd.DataFrame:
    """마스터 CSV를 로드하고 결측치/Total 행을 방어적으로 정제한다."""
    df = pd.read_csv(csv_path)

    # Total 잔재 행 제거
    if "month" in df.columns:
        df = df[df["month"].astype(str).str.strip().str.lower() != "total"]

    # 금액 정제: 숫자 변환 후 0 이하/Null 제외
    df["Amount_KRW"] = pd.to_numeric(df["Amount_KRW"], errors="coerce")
    df = df[df["Amount_KRW"].notna() & (df["Amount_KRW"] > 0)]

    # 결제일시 파싱 및 파생 컬럼
    df["Approved_DateTime"] = pd.to_datetime(df["Approved_DateTime"], errors="coerce")
    df["hour"] = df["Approved_DateTime"].dt.hour
    df["weekday"] = df["Approved_DateTime"].dt.weekday.map(
        {i: WEEKDAY_KR[i] for i in range(7)}
    )

    return df.reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    months: list[str],
    categories: list[str],
    cards: list[str],
    stores: list[str],
    amount_range: tuple[int, int],
) -> pd.DataFrame:
    """사이드바 조건을 DataFrame에 적용한다."""
    out = df.copy()
    if months:
        out = out[out["month"].isin(months)]
    if categories:
        out = out[out["Category"].isin(categories)]
    if cards:
        out = out[out["Card_Name"].isin(cards)]
    if stores:
        out = out[out["Store_Name"].isin(stores)]
    lo, hi = amount_range
    out = out[(out["Amount_KRW"] >= lo) & (out["Amount_KRW"] <= hi)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
def calculate_kpi(df: pd.DataFrame) -> dict:
    """토스형 핵심 재무 지표를 계산한다."""
    total = int(df["Amount_KRW"].sum())
    avg = int(df["Amount_KRW"].mean()) if len(df) else 0

    cat_sum = df.groupby("Category")["Amount_KRW"].sum()
    top_cat = cat_sum.idxmax() if not cat_sum.empty else "-"
    top_cat_amt = int(cat_sum.max()) if not cat_sum.empty else 0

    max_row = df.loc[df["Amount_KRW"].idxmax()] if len(df) else None
    max_amt = int(max_row["Amount_KRW"]) if max_row is not None else 0
    max_store = max_row["Store_Name"] if max_row is not None else "-"

    cat_std = df.groupby("Category")["Amount_KRW"].std().fillna(0)
    volatile_cat = cat_std.idxmax() if not cat_std.empty else "-"
    volatile_std = int(cat_std.max()) if not cat_std.empty else 0

    return {
        "total": total,
        "avg": avg,
        "top_cat": top_cat,
        "top_cat_amt": top_cat_amt,
        "max_amt": max_amt,
        "max_store": max_store,
        "volatile_cat": volatile_cat,
        "volatile_std": volatile_std,
    }


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def draw_trend_charts(df: pd.DataFrame, group_by: str) -> None:
    """(1) 월별 지출 동향 및 추이."""
    trend = df.groupby(["month", group_by], as_index=False)["Amount_KRW"].sum()
    fig = px.area(
        trend,
        x="month",
        y="Amount_KRW",
        color=group_by,
        markers=True,
        title=f"월별 지출 추이 ({group_by} 기준)",
        labels={"month": "월", "Amount_KRW": "지출 합계(원)", group_by: group_by},
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def draw_compare_charts(df: pd.DataFrame) -> None:
    """(2) 카드별/카테고리별 소비 효율 비교."""
    c1, c2 = st.columns(2)
    with c1:
        card_sum = df.groupby(["Card_Name", "Category"], as_index=False)["Amount_KRW"].sum()
        fig = px.bar(
            card_sum,
            x="Card_Name",
            y="Amount_KRW",
            color="Category",
            title="카드별 지출 규모 (카테고리 누적)",
            labels={"Card_Name": "카드", "Amount_KRW": "지출 합계(원)"},
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        cat_sum = df.groupby("Category", as_index=False)["Amount_KRW"].sum().sort_values("Amount_KRW")
        fig = px.bar(
            cat_sum,
            x="Amount_KRW",
            y="Category",
            orientation="h",
            title="카테고리별 총 지출",
            labels={"Category": "카테고리", "Amount_KRW": "지출 합계(원)"},
            color="Category",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")


def draw_distribution_charts(df: pd.DataFrame) -> None:
    """(3) 소비 이상치 및 지출 변동성 (Box Plot)."""
    fig = px.box(
        df,
        x="Category",
        y="Amount_KRW",
        color="Category",
        points="outliers",
        title="카테고리별 지출 분포 및 이상치(Outlier)",
        labels={"Category": "카테고리", "Amount_KRW": "결제 금액(원)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


def draw_heatmap(df: pd.DataFrame) -> None:
    """(4) 요일별/카테고리별 소비 구조 히트맵."""
    if df["weekday"].isna().all():
        st.info("결제일시 정보가 없어 히트맵을 표시할 수 없습니다.")
        return
    pivot = df.pivot_table(
        index="Category",
        columns="weekday",
        values="Amount_KRW",
        aggfunc="mean",
        fill_value=0,
    )
    # 요일 순서 정렬
    ordered = [d for d in WEEKDAY_KR if d in pivot.columns]
    pivot = pivot[ordered]
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="Oranges",
        title="요일 × 카테고리 평균 지출액 히트맵",
        labels=dict(x="요일", y="카테고리", color="평균 지출(원)"),
    )
    st.plotly_chart(fig, width="stretch")


def draw_top_items(df: pd.DataFrame, top_n: int = 10) -> None:
    """(5) 과소비 주범 Top-N 가맹점."""
    store_sum = (
        df.groupby("Store_Name", as_index=False)["Amount_KRW"]
        .sum()
        .sort_values("Amount_KRW", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        store_sum.sort_values("Amount_KRW"),
        x="Amount_KRW",
        y="Store_Name",
        orientation="h",
        title=f"과소비 주범 Top {top_n} 가맹점",
        labels={"Store_Name": "가맹점", "Amount_KRW": "지출 합계(원)"},
        text="Amount_KRW",
    )
    fig.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 자동 요약 리포트
# ---------------------------------------------------------------------------
def build_summary_report(df: pd.DataFrame, kpi: dict) -> str:
    """규칙 기반 소비 동향 요약 문장을 생성한다."""
    months = sorted(df["month"].unique())
    period = f"{months[0]}부터 {months[-1]}까지" if months else "선택 기간"

    # 야간(0~5시) 배달/식비 변동성
    night = df[(df["hour"] >= 0) & (df["hour"] <= 5)]
    night_note = ""
    if not night.empty:
        night_top = (
            night.groupby("Store_Name")["Amount_KRW"].sum().sort_values(ascending=False).head(2)
        )
        if not night_top.empty:
            stores = ", ".join(night_top.index.tolist())
            night_note = f" 특히 {stores}에서의 야간(0~5시) 지출 변동성이 크게 나타났습니다."

    lines = (
        f"{period} 사용자의 누적 지출액은 총 {kpi['total']:,}원이며, "
        f"이 중 가장 큰 비중을 차지하는 과소비 주범은 '{kpi['top_cat']}({kpi['top_cat_amt']:,}원)'입니다. "
        f"건당 평균 결제 금액은 {kpi['avg']:,}원이고, 단일 최고 지출은 "
        f"{kpi['max_store']}에서 발생한 {kpi['max_amt']:,}원입니다. "
        f"지출 통제가 가장 필요한(변동성 최대) 카테고리는 '{kpi['volatile_cat']}'입니다."
        f"{night_note} "
        f"지출 통제를 위해 다음 달 '{kpi['top_cat']}' 지출을 10% 줄이는 챌린지에 도전할 것을 권장합니다."
    )
    return lines


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    """사이드바 인터랙티브 필터를 렌더링한다."""
    st.sidebar.header("🔍 실시간 필터")

    months = sorted(df["month"].unique())
    categories = sorted(df["Category"].unique())
    cards = sorted(df["Card_Name"].unique())
    stores = sorted(df["Store_Name"].unique())
    amt_min, amt_max = int(df["Amount_KRW"].min()), int(df["Amount_KRW"].max())

    sel_months = st.sidebar.multiselect("조회 기간 (month)", months, default=months)
    sel_cats = st.sidebar.multiselect("소비 분류 (Category)", categories, default=categories)
    sel_cards = st.sidebar.multiselect("보유 카드 (Card_Name)", cards, default=cards)
    sel_stores = st.sidebar.multiselect(
        "자주 가는 가맹점 (Store_Name) — 미선택 시 전체", stores, default=[]
    )
    sel_amount = st.sidebar.slider(
        "결제 금액 범위 (원)",
        min_value=amt_min,
        max_value=amt_max,
        value=(amt_min, amt_max),
        step=1000,
    )

    st.sidebar.divider()
    st.sidebar.caption(f"전체 데이터: {len(df):,}건")

    return sel_months, sel_cats, sel_cards, sel_stores, sel_amount


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="가계부 분석 대시보드", page_icon="💳", layout="wide")
    st.title("💳 개인 맞춤형 자산 관리 가계부 분석 대시보드")
    st.caption("마이데이터 기반 소비 성향 탐색 · 비교 · 진단 PFM 시스템")

    # 데이터 로드
    if not CSV_PATH.exists():
        st.error(
            f"데이터 파일을 찾을 수 없습니다: {CSV_PATH}\n\n"
            "먼저 merge.py를 실행해 user_card_history_merged.csv를 생성하세요."
        )
        st.stop()

    try:
        df = load_data(str(CSV_PATH))
    except Exception as exc:  # noqa: BLE001
        st.error(f"데이터 로드 중 오류가 발생했습니다: {exc}")
        st.stop()

    if df.empty:
        st.warning("유효한 가계부 데이터가 없습니다.")
        st.stop()

    st.caption(f"총 누적 데이터: {len(df):,}건 로드 완료")

    # 필터
    sel_months, sel_cats, sel_cards, sel_stores, sel_amount = render_sidebar(df)
    filtered = apply_filters(df, sel_months, sel_cats, sel_cards, sel_stores, sel_amount)

    if filtered.empty:
        st.warning("조건에 맞는 가계부 내역이 없습니다.")
        st.stop()

    # EDA 요약
    st.subheader("📋 데이터 탐색 요약")
    e1, e2, e3 = st.columns(3)
    e1.metric("총 지출 건수", f"{len(filtered):,}건")
    e2.metric("고유 가맹점 수", f"{filtered['Store_Name'].nunique():,}곳")
    e3.metric("분석 대상 카테고리", f"{filtered['Category'].nunique():,}개")

    # KPI
    st.subheader("💰 핵심 재무 KPI")
    kpi = calculate_kpi(filtered)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("총 소비 금액", f"{kpi['total']:,}원")
    k2.metric("평균 결제 금액", f"{kpi['avg']:,}원")
    k3.metric("최다 소비 카테고리", kpi["top_cat"], f"{kpi['top_cat_amt']:,}원")
    k4.metric("최고 지출", f"{kpi['max_amt']:,}원", kpi["max_store"])
    k5.metric("최대 변동성 카테고리", kpi["volatile_cat"], f"σ {kpi['volatile_std']:,}")

    # 데이터 테이블
    with st.expander(f"필터링된 가계부 내역 보기 ({len(filtered):,}건)"):
        show = filtered.copy()
        show["Approved_DateTime"] = show["Approved_DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            show[
                ["month", "Approved_DateTime", "Card_Name", "Store_Name", "Category", "Amount_KRW"]
            ],
            width="stretch",
            height=320,
        )

    # 시각화
    st.subheader("📊 소비 분석 시각화")
    t1, t2, t3, t4, t5 = st.tabs(
        ["월별 추이", "카드/카테고리 비교", "지출 변동성", "요일 히트맵", "Top 가맹점"]
    )
    with t1:
        group_by = st.radio(
            "그룹 기준", ["Category", "Card_Name"], horizontal=True, key="trend_group"
        )
        draw_trend_charts(filtered, group_by)
    with t2:
        draw_compare_charts(filtered)
    with t3:
        draw_distribution_charts(filtered)
    with t4:
        draw_heatmap(filtered)
    with t5:
        top_n = st.slider("Top N", 5, 15, 10, key="topn")
        draw_top_items(filtered, top_n)

    # 자동 요약 리포트
    st.subheader("📝 마이데이터 기반 소비 동향 자동 요약")
    st.info(build_summary_report(filtered, kpi))


if __name__ == "__main__":
    main()
