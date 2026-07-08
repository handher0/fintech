"""
금융위 API 기반 상장종목 시계열 주가 분석 시스템

공공데이터포털 금융위원회 KRX 상장종목정보·주식시세정보 API를 연동하여
종목 검색, 이동평균선, 조건 필터, Plotly 복합 차트를 제공한다.

실행 방법:
  pip install streamlit plotly python-dotenv pandas requests
  cd fintech/8.FSCcmp
  streamlit run code/stock_timeseries.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
LOGO_PATH = PROJECT_ROOT / "logo.png"
CHARTS_DIR = CODE_DIR.parent / "output" / "charts"

LISTED_URL = "http://apis.data.go.kr/1160100/service/GetKrxListedInfoService/getItemInfo"
PRICE_URL = "http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"

BEGIN_BAS_DT = "20260101"
DEFAULT_STOCK_NAME = "삼성전자"
DEFAULT_STOCK_CODE = "005930"

HEADER_CSS = """
<style>
.fsc-header-strip {
    height: 7px;
    width: 100%;
    background: linear-gradient(to right, #FF4B4B, #1C65E3);
    border-radius: 4px;
    margin-bottom: 1rem;
}
</style>
<div class="fsc-header-strip"></div>
"""


# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------
def _parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return parsed
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def load_api_key() -> tuple[str | None, Path]:
    merged: dict[str, str] = {}
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        merged.update(_parse_env_file(ENV_PATH))
    key = (os.getenv("PUBLIC_DATA_API_KEY") or merged.get("PUBLIC_DATA_API_KEY") or "").strip()
    return (key or None, ENV_PATH)


# ---------------------------------------------------------------------------
# API 공통
# ---------------------------------------------------------------------------
def _build_url(base: str, api_key: str, params: dict[str, Any]) -> str:
    """serviceKey는 URL에 직접 바인딩하여 이중 인코딩을 방지한다."""
    query = urlencode({k: v for k, v in params.items() if v is not None})
    return f"{base}?serviceKey={api_key}&{query}"


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    header = (payload.get("response") or {}).get("header") or {}
    code = str(header.get("resultCode", ""))
    if code not in ("00", "0", ""):
        msg = header.get("resultMsg", "API 오류")
        raise RuntimeError(f"공공데이터 API 오류 ({code}): {msg}")

    body = (payload.get("response") or {}).get("body") or {}
    items = (body.get("items") or {})
    if not items:
        return []
    item = items.get("item")
    if item is None:
        return []
    if isinstance(item, dict):
        return [item]
    return list(item)


def _normalize_stock_code(code: str) -> str:
    """KRX 단축코드(A005930)를 시세 API 조회용(005930)으로 정규화한다."""
    code = str(code).strip()
    if code.upper().startswith("A") and len(code) > 1:
        return code[1:]
    return code


def _api_get(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# KRX 상장종목 마스터
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="KRX 상장종목 마스터 수집 중...")
def fetch_listed_master(api_key: str) -> pd.DataFrame:
    """최신 basDt 기준 상장종목 마스터를 페이지네이션으로 수집한다."""
    probe_url = _build_url(LISTED_URL, api_key, {"pageNo": 1, "numOfRows": 1, "resultType": "json"})
    probe = _api_get(probe_url)
    probe_rows = _extract_items(probe)
    if not probe_rows:
        return pd.DataFrame()

    bas_dt = str(probe_rows[0].get("basDt", "")).strip()
    if not bas_dt:
        raise RuntimeError("최신 basDt를 확인할 수 없습니다.")

    all_rows: list[dict[str, Any]] = []
    page = 1
    page_size = 1000
    while True:
        url = _build_url(
            LISTED_URL,
            api_key,
            {"pageNo": page, "numOfRows": page_size, "resultType": "json", "basDt": bas_dt},
        )
        rows = _extract_items(_api_get(url))
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["label"] = df["itmsNm"].astype(str) + " (" + df["srtnCd"].astype(str) + ")"
    return df.sort_values("itmsNm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 주식 시세
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="주가 시계열 수집 중...")
def fetch_stock_prices(
    api_key: str,
    stock_name: str,
    stock_code: str,
    begin: str,
    end: str,
) -> pd.DataFrame:
    """종목별 주가 시계열을 페이지네이션으로 수집한다."""
    query_code = _normalize_stock_code(stock_code)
    all_rows: list[dict[str, Any]] = []
    page = 1
    page_size = 1000

    while True:
        url = _build_url(
            PRICE_URL,
            api_key,
            {
                "pageNo": page,
                "numOfRows": page_size,
                "resultType": "json",
                "beginBasDt": begin,
                "endBasDt": end,
                "likeItmsNm": stock_name,
                "likeSrtnCd": query_code,
            },
        )
        rows = _extract_items(_api_get(url))
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    numeric_cols = ["clpr", "vs", "fltRt", "hipr", "lopr", "trqu"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DATE"] = pd.to_datetime(df["basDt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """5·20·60일 이동평균선(SMA)을 추가한다."""
    out = df.copy()
    if "clpr" not in out.columns:
        return out
    out["SMA5"] = out["clpr"].rolling(5, min_periods=1).mean()
    out["SMA20"] = out["clpr"].rolling(20, min_periods=1).mean()
    out["SMA60"] = out["clpr"].rolling(60, min_periods=1).mean()
    return out


def apply_filters(
    df: pd.DataFrame,
    min_volume: int,
    ma_filter: str,
) -> pd.DataFrame:
    """거래량·이동평균 정배열/역배열 조건으로 필터링한다."""
    if df.empty:
        return df
    out = df.copy()
    if min_volume > 0 and "trqu" in out.columns:
        out = out[out["trqu"] >= min_volume]

    if ma_filter == "정배열 (SMA5 > SMA20 > SMA60)":
        mask = (out["SMA5"] > out["SMA20"]) & (out["SMA20"] > out["SMA60"])
        out = out[mask]
    elif ma_filter == "역배열 (SMA5 < SMA20 < SMA60)":
        mask = (out["SMA5"] < out["SMA20"]) & (out["SMA20"] < out["SMA60"])
        out = out[mask]
    return out.reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev_close = float(df.iloc[-2]["clpr"]) if len(df) >= 2 else float(latest["clpr"])
    current = float(latest["clpr"])
    delta = current - prev_close
    volume = int(latest["trqu"]) if pd.notna(latest.get("trqu")) else 0
    period_high = float(df["hipr"].max()) if "hipr" in df.columns else current
    pct_from_high = ((current / period_high) - 1) * 100 if period_high else 0.0
    return {
        "close": current,
        "delta": delta,
        "volume": volume,
        "pct_from_high": pct_from_high,
        "as_of": latest["DATE"],
    }


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def build_price_volume_chart(df: pd.DataFrame, stock_label: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{stock_label} 종가 및 이동평균선", "거래량"),
    )
    fig.add_trace(
        go.Scatter(x=df["DATE"], y=df["clpr"], name="종가", mode="lines", line={"width": 2}),
        row=1, col=1,
    )
    for col, name, color in [
        ("SMA5", "SMA5", "#f39c12"),
        ("SMA20", "SMA20", "#9b59b6"),
        ("SMA60", "SMA60", "#27ae60"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["DATE"], y=df[col], name=name, mode="lines", line={"width": 1.2, "color": color}),
                row=1, col=1,
            )
    prev_close = None
    colors = []
    for _, row in df.iterrows():
        if prev_close is None:
            colors.append("#95a5a6")
        else:
            colors.append("#e74c3c" if row["clpr"] >= prev_close else "#3498db")
        prev_close = row["clpr"]
    fig.add_trace(
        go.Bar(x=df["DATE"], y=df["trqu"], name="거래량", marker_color=colors, opacity=0.7),
        row=2, col=1,
    )
    fig.update_layout(height=620, hovermode="x unified", legend={"orientation": "h", "y": 1.08})
    fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=-30, nticks=10, row=2, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=-30, nticks=10, row=1, col=1)
    return fig


def save_chart_html(fig: go.Figure, stock_code: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CHARTS_DIR / f"stock_{stock_code}_{ts}.html"
    fig.write_html(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# UI 헬퍼
# ---------------------------------------------------------------------------
def filter_master(master: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return master
    q = query.strip().lower()
    mask = (
        master["itmsNm"].astype(str).str.lower().str.contains(q, na=False)
        | master["srtnCd"].astype(str).str.lower().str.contains(q, na=False)
        | master["srtnCd"].astype(str).apply(_normalize_stock_code).str.contains(q, na=False)
    )
    return master[mask].reset_index(drop=True)


def default_stock_index(labels: list[str]) -> int:
    for i, lbl in enumerate(labels):
        if DEFAULT_STOCK_NAME in lbl or DEFAULT_STOCK_CODE in lbl:
            return i
    return 0


def init_session() -> None:
    if "stock_search" not in st.session_state:
        st.session_state.stock_search = ""
    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None


def reset_screen() -> None:
    st.session_state.stock_search = ""
    st.session_state.selected_label = None
    st.session_state.pop("chart_saved_msg", None)


def render_logo() -> None:
    try:
        if LOGO_PATH.exists():
            st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="상장종목 시계열 주가 분석", page_icon="📈", layout="wide")
    init_session()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    render_logo()

    api_key, env_path = load_api_key()
    with st.sidebar:
        st.header("종목 검색")
        st.session_state.stock_search = st.text_input(
            "종목명 · 단축코드 검색",
            value=st.session_state.stock_search,
            placeholder='예: "삼성", "NAVER", "005930"',
        )

    if not api_key:
        st.error(
            "API Key가 로드되지 않았습니다. 상위 fintech/.env 파일의 "
            "PUBLIC_DATA_API_KEY 변수명과 경로를 재확인해 주세요."
        )
        st.caption(f"확인한 .env 경로: {env_path}")
        return

    st.markdown(HEADER_CSS, unsafe_allow_html=True)
    st.title("📈 금융위 API 기반 상장종목 시계열 주가 분석 시스템")

    try:
        master = fetch_listed_master(api_key)
    except Exception as exc:
        st.error(f"상장종목 마스터 수집 실패: {exc}")
        return

    if master.empty:
        st.error("상장종목 데이터를 불러오지 못했습니다.")
        return

    filtered = filter_master(master, st.session_state.stock_search)
    if filtered.empty:
        st.warning("검색 결과가 없습니다. 검색어를 변경해 주세요.")
        filtered = master

    labels = filtered["label"].tolist()
    with st.sidebar:
        st.caption(f"검색 결과 {len(filtered):,}개 / 전체 {len(master):,}개")
        default_idx = default_stock_index(labels)
        if st.session_state.selected_label in labels:
            default_idx = labels.index(st.session_state.selected_label)
        selected_label = st.selectbox("종목 선택", labels, index=default_idx)
        st.session_state.selected_label = selected_label

        st.divider()
        st.header("제어 패널")
        if st.button("화면 초기화 (Reset)", use_container_width=True):
            reset_screen()
            st.rerun()

    row = filtered[filtered["label"] == selected_label].iloc[0]
    stock_name = str(row["itmsNm"])
    stock_code = str(row["srtnCd"])

    end_dt = datetime.now().strftime("%Y%m%d")
    try:
        prices = fetch_stock_prices(api_key, stock_name, stock_code, BEGIN_BAS_DT, end_dt)
    except Exception as exc:
        st.error(f"주가 데이터 수집 실패: {exc}")
        return

    if prices.empty:
        st.warning(f"{stock_name}({stock_code})의 주가 데이터가 없습니다.")
        return

    prices = add_technical_indicators(prices)
    kpis = compute_kpis(prices)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "현재 종가",
            f"{kpis['close']:,.0f}원",
            delta=f"{kpis['delta']:+,.0f}원",
        )
    with c2:
        st.metric("당일 거래량", f"{kpis['volume']:,}주")
    with c3:
        st.metric(
            "기간 최고가 대비",
            f"{kpis['pct_from_high']:+.2f}%",
            help="조회 구간 내 최고가(hipr) 대비 현재 종가 위치",
        )
    st.caption(f"기준일: {kpis['as_of'].strftime('%Y-%m-%d')}")

    st.divider()
    st.subheader("데이터 필터링")
    fc1, fc2 = st.columns(2)
    with fc1:
        max_vol = int(prices["trqu"].max()) if "trqu" in prices.columns else 0
        min_volume = st.slider(
            "최소 거래량 (주)",
            min_value=0,
            max_value=max(max_vol, 1),
            value=0,
            step=max(max_vol // 100, 1) if max_vol > 0 else 1,
        )
    with fc2:
        ma_filter = st.radio(
            "이동평균선 조건",
            ["전체", "정배열 (SMA5 > SMA20 > SMA60)", "역배열 (SMA5 < SMA20 < SMA60)"],
            horizontal=True,
        )

    filtered_prices = apply_filters(prices, min_volume, ma_filter)

    st.subheader("주가 동향 및 거래량")
    chart_df = filtered_prices if not filtered_prices.empty else prices
    fig = build_price_volume_chart(chart_df, f"{stock_name} ({stock_code})")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("보고서용 차트 저장 (HTML)", use_container_width=True):
        try:
            saved = save_chart_html(fig, stock_code)
            st.session_state.chart_saved_msg = saved
            st.success(f"차트 저장 완료: {saved}")
        except Exception as exc:
            st.error(f"차트 저장 실패: {exc}")

    if st.session_state.get("chart_saved_msg"):
        st.caption(f"최근 저장: {st.session_state.chart_saved_msg}")

    st.subheader("시계열 데이터")
    display_cols = ["DATE", "basDt", "clpr", "vs", "fltRt", "trqu", "SMA5", "SMA20", "SMA60", "hipr", "lopr"]
    show = chart_df[[c for c in display_cols if c in chart_df.columns]].copy()
    if "DATE" in show.columns:
        show["DATE"] = show["DATE"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
