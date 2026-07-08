"""
API 기반 환율 및 거시경제 지표 AI 분석 에이전트

한국은행 ECOS Open API로 환율·기준금리·CPI를 수집하고,
Streamlit KPI 대시보드 + Plotly 차트 + OpenAI 챗봇을 제공한다.

실행 방법:
  pip install streamlit plotly matplotlib seaborn python-dotenv openai pandas requests
  cd 7.BoKMacro
  streamlit run code/macro_data.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
CHARTS_DIR = BASE_DIR / "output" / "charts"

ECOS_BASE_URL = "http://ecos.bok.or.kr/api/StatisticSearch"

DISPLAY_START_D = "20260101"
DISPLAY_START_M = "202601"
CPI_FETCH_START_M = "202501"

INDICATOR_CONFIGS = {
    "exchange": {
        "label": "원/달러 환율",
        "primary": {"stat_code": "022Y013", "cycle": "D", "item_code": None},
        "fallback": {"stat_code": "731Y001", "cycle": "D", "item_code": "0000001"},
        "unit": "원",
        "chart_key": "exchange_rate",
    },
    "rate": {
        "label": "한국은행 기준금리",
        "primary": {"stat_code": "098Y001", "cycle": "M", "item_code": None},
        "fallback": {"stat_code": "722Y001", "cycle": "D", "item_code": "0101000"},
        "unit": "%",
        "chart_key": "base_rate",
    },
    "cpi": {
        "label": "소비자물가지수(CPI)",
        "primary": {"stat_code": "021Y125", "cycle": "M", "item_code": None},
        "fallback": {"stat_code": "901Y009", "cycle": "M", "item_code": "0"},
        "unit": "지수",
        "chart_key": "cpi",
    },
}

MACRO_SYSTEM_PROMPT = """너는 거시경제 지표를 명확하고 거품 없이 분석해 주는 전문 매크로 이코노미스트 비서다.

답변 형식 규칙 (반드시 준수):
- 마크다운 기호(#, ##, **, `, >, ---)를 절대 사용하지 마세요.
- 일반 문단 텍스트와 불렛 리스트(-)만 사용하세요.
- 수치는 제공된 데이터 컨텍스트를 우선 인용하세요.
- web_search 참고 자료가 있으면 서두에 핵심 숫자를 명시하세요.
- 추측보다 데이터에 기반해 답변하세요.
"""


# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------
def _parse_env_file(path: Path) -> dict[str, str]:
    """utf-8-sig 인코딩으로 KEY=VALUE 형식을 직접 파싱한다."""
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


def load_merged_env() -> tuple[dict[str, str], list[Path]]:
    """7.BoKMacro/.env → fintech/.env 순으로 병합 로드한다."""
    merged: dict[str, str] = {}
    checked: list[Path] = []
    for path in (BASE_DIR / ".env", PROJECT_ROOT / ".env"):
        if not path.exists():
            continue
        checked.append(path.resolve())
        load_dotenv(dotenv_path=path, override=True)
        merged.update(_parse_env_file(path))
    return merged, checked


def resolve_api_keys(merged: dict[str, str]) -> tuple[str | None, str | None, list[str]]:
    """ECOS_API_KEY, OPENAI_API_KEY를 os.getenv와 merged에서 병합 조회한다."""
    missing: list[str] = []

    ecos = (os.getenv("ECOS_API_KEY") or merged.get("ECOS_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or merged.get("OPENAI_API_KEY") or "").strip()

    if not ecos:
        missing.append("ECOS_API_KEY")
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    return (ecos or None, openai_key or None, missing)


def show_key_error(missing: list[str], checked_paths: list[Path]) -> None:
    st.error(
        "API Key가 로드되지 않았습니다. .env 파일의 변수명(ECOS_API_KEY)과 파일 경로를 재확인해 주세요."
    )
    st.caption(f"누락된 변수: {', '.join(missing)}")
    if checked_paths:
        st.caption("확인한 .env 경로: " + " | ".join(str(p) for p in checked_paths))
    else:
        st.caption(
            f"확인한 .env 경로: {BASE_DIR / '.env'} | {PROJECT_ROOT / '.env'} (파일 없음)"
        )


# ---------------------------------------------------------------------------
# ECOS API
# ---------------------------------------------------------------------------
def _end_date_for_cycle(cycle: str) -> str:
    today = datetime.now()
    return today.strftime("%Y%m%d") if cycle == "D" else today.strftime("%Y%m")


def _build_ecos_url(
    api_key: str,
    stat_code: str,
    cycle: str,
    start: str,
    end: str,
    start_idx: int,
    end_idx: int,
    item_code: str | None = None,
) -> str:
    base = (
        f"{ECOS_BASE_URL}/{api_key}/json/kr/{start_idx}/{end_idx}"
        f"/{stat_code}/{cycle}/{start}/{end}"
    )
    if item_code is not None:
        return f"{base}/{item_code}"
    return base


def _parse_ecos_response(payload: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    """ECOS JSON 응답을 DataFrame으로 변환하고 오류 코드를 반환한다."""
    if "RESULT" in payload:
        code = str(payload["RESULT"].get("CODE", ""))
        if code == "INFO-200":
            return pd.DataFrame(), code
        if code and code != "INFO-000":
            return pd.DataFrame(), code

    block = payload.get("StatisticSearch") or {}
    rows = block.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return pd.DataFrame(), "INFO-200"

    df = pd.DataFrame(rows)
    if "DATA_VALUE" in df.columns:
        df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df = df.dropna(subset=["DATA_VALUE"])
    return df.reset_index(drop=True), None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ecos_cached(
    api_key: str,
    stat_code: str,
    cycle: str,
    start: str,
    end: str,
    item_code: str | None,
) -> pd.DataFrame:
    """ECOS StatisticSearch API를 페이지네이션으로 호출한다 (캐시 대상)."""
    all_rows: list[dict[str, Any]] = []
    page_size = 100
    start_idx = 1

    while True:
        end_idx = start_idx + page_size - 1
        url = _build_ecos_url(
            api_key, stat_code, cycle, start, end, start_idx, end_idx, item_code
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break

        df_page, err_code = _parse_ecos_response(payload)
        if err_code == "INFO-200" and not all_rows:
            return pd.DataFrame()
        if err_code and err_code != "INFO-200":
            break
        if df_page.empty:
            break

        all_rows.extend(df_page.to_dict("records"))
        if len(df_page) < page_size:
            break
        start_idx += page_size

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).reset_index(drop=True)


def _date_range_for_cycle(
    cycle: str,
    start_d: str,
    end_d: str,
    start_m: str,
    end_m: str,
) -> tuple[str, str]:
    """주기(D/M)에 맞는 ECOS 조회 시작·종료 문자열을 반환한다."""
    if cycle == "D":
        return start_d, end_d
    return start_m, end_m


def fetch_indicator(
    api_key: str,
    config: dict[str, Any],
    start_d: str,
    end_d: str,
    start_m: str,
    end_m: str,
) -> tuple[pd.DataFrame, str]:
    """1차 통계코드 실패 시 폴백으로 재조회한다 (캐시 바깥)."""
    for key in ("primary", "fallback"):
        spec = config[key]
        cycle = spec["cycle"]
        start, end = _date_range_for_cycle(cycle, start_d, end_d, start_m, end_m)
        item = spec.get("item_code")
        df = _fetch_ecos_cached(
            api_key,
            spec["stat_code"],
            cycle,
            start,
            end,
            item,
        )
        if not df.empty:
            df = _prepare_series(df, cycle)
            return df, cycle
    return pd.DataFrame(), config["primary"]["cycle"]


def _prepare_series(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """TIME → DATE 변환 및 VALUE 컬럼 정리."""
    out = df.copy()
    out["TIME"] = out["TIME"].astype(str)
    out["VALUE"] = out["DATA_VALUE"].astype(float)
    out["DATE"] = out["TIME"].apply(lambda t: _parse_time(t, cycle))
    out = out.sort_values("DATE").reset_index(drop=True)
    return out


def _parse_time(time_str: str, cycle: str) -> pd.Timestamp:
    s = str(time_str).strip()
    if cycle == "D":
        return pd.to_datetime(s, format="%Y%m%d")
    return pd.to_datetime(s + "01", format="%Y%m%d")


def filter_display_period(df: pd.DataFrame) -> pd.DataFrame:
    """차트·챗봇 컨텍스트용 2026년 이후 데이터만 반환한다."""
    if df.empty:
        return df
    cutoff = pd.Timestamp("2026-01-01")
    return df[df["DATE"] >= cutoff].reset_index(drop=True)


def compute_cpi_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """CPI 지수로 전년동월비(YoY %)를 산출한다."""
    out = df.sort_values("DATE").copy()
    out["YoY"] = out["VALUE"].pct_change(periods=12) * 100
    return out


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
def kpi_exchange(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 1:
        return {"value": None, "delta": None, "as_of": None}
    latest = df.iloc[-1]
    delta = None
    if len(df) >= 2:
        delta = float(latest["VALUE"] - df.iloc[-2]["VALUE"])
    return {
        "value": float(latest["VALUE"]),
        "delta": delta,
        "as_of": latest["DATE"],
    }


def kpi_rate(df: pd.DataFrame) -> dict[str, Any]:
    """직전 변경 시점 대비 증감(%p) — step_series 로직."""
    if df.empty:
        return {"value": None, "delta": None, "as_of": None}
    latest_row = df.iloc[-1]
    changes = df[df["VALUE"] != df["VALUE"].shift(1)].reset_index(drop=True)
    delta = None
    if len(changes) >= 2:
        delta = float(changes.iloc[-1]["VALUE"] - changes.iloc[-2]["VALUE"])
    return {
        "value": float(latest_row["VALUE"]),
        "delta": delta,
        "as_of": latest_row["DATE"],
    }


def kpi_cpi_yoy(df: pd.DataFrame) -> dict[str, Any]:
    """CPI 전년동월비 상승률 KPI — 전월 YoY 대비 델타."""
    enriched = compute_cpi_yoy(df)
    valid = enriched.dropna(subset=["YoY"])
    display = filter_display_period(valid)
    if display.empty:
        return {"value": None, "delta": None, "as_of": None}
    latest = display.iloc[-1]
    delta = None
    if len(display) >= 2:
        delta = float(latest["YoY"] - display.iloc[-2]["YoY"])
    return {
        "value": float(latest["YoY"]),
        "delta": delta,
        "as_of": latest["DATE"],
    }


def format_as_of(dt: pd.Timestamp | None, cycle: str) -> str:
    if dt is None or pd.isna(dt):
        return "-"
    if cycle == "D":
        return f"기준일: {dt.strftime('%Y-%m-%d')}"
    return f"기준월: {dt.strftime('%Y-%m')}"


# ---------------------------------------------------------------------------
# LLM & 텍스트
# ---------------------------------------------------------------------------
def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def build_data_context(
    exchange_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
) -> str:
    """챗봇에 전달할 거시경제 데이터 컨텍스트 문자열."""
    parts: list[str] = []

    if not exchange_df.empty:
        ex = exchange_df.tail(10)[["DATE", "VALUE"]].copy()
        ex["DATE"] = ex["DATE"].dt.strftime("%Y-%m-%d")
        parts.append("[원/달러 환율 (최근 10일)]\n" + ex.to_string(index=False))

    if not rate_df.empty:
        rt = rate_df.tail(12)[["DATE", "VALUE"]].copy()
        rt["DATE"] = rt["DATE"].dt.strftime("%Y-%m")
        parts.append("[한국은행 기준금리 (최근 12개월)]\n" + rt.to_string(index=False))

    if not cpi_df.empty:
        cpi_enriched = compute_cpi_yoy(cpi_df)
        cpi_disp = filter_display_period(cpi_enriched).tail(12)
        if not cpi_disp.empty:
            cp = cpi_disp[["DATE", "VALUE", "YoY"]].copy()
            cp["DATE"] = cp["DATE"].dt.strftime("%Y-%m")
            parts.append("[CPI 지수 및 전년동월비 YoY (최근 12개월)]\n" + cp.to_string(index=False))

    ex_kpi = kpi_exchange(exchange_df)
    rt_kpi = kpi_rate(rate_df)
    cp_kpi = kpi_cpi_yoy(cpi_df)
    parts.append(
        "[현재 KPI 요약]\n"
        f"- 환율: {ex_kpi['value']}원 (전일 대비 {ex_kpi['delta']}원)\n"
        f"- 기준금리: {rt_kpi['value']}% (직전 변경 대비 {rt_kpi['delta']}%p)\n"
        f"- CPI YoY: {cp_kpi['value']}% (전월 대비 {cp_kpi['delta']}%p)"
    )
    return "\n\n".join(parts)


def call_llm(client: OpenAI, model: str, question: str, context: str, web_context: str = "") -> str:
    extra = f"\n\n[웹 검색 참고]\n{web_context}" if web_context else ""
    user_msg = f"[거시경제 데이터]\n{context}{extra}\n\n[사용자 질문]\n{question}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MACRO_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _strip_markdown(raw)
    except Exception as exc:
        return f"OpenAI API 호출 중 오류가 발생했습니다: {exc}"


def needs_global_web_search(question: str) -> bool:
    keywords = (
        "글로벌", "미국", "연준", "fed", "세계", "국제", "유럽", "중국",
        "일본", "영국", "ecb", "인플레", "글로벌 경제", "세계 경제",
    )
    q = question.lower()
    return any(kw in q for kw in keywords)


def run_web_search(client: OpenAI, question: str) -> str:
    """OpenAI Responses API web_search로 글로벌 금융 트렌드를 조회한다."""
    if not hasattr(client, "responses"):
        return ""
    prompt = (
        "2026년 현재 글로벌 거시경제·금융 트렌드 관점에서 다음 질문에 답하세요. "
        "서두에 핵심 수치(금리, 물가, 환율 등)를 명시하세요.\n\n"
        f"질문: {question}"
    )
    for tool_type in ("web_search", "web_search_preview"):
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                tools=[{"type": tool_type}],
            )
            if getattr(resp, "output_text", ""):
                return _strip_markdown(str(resp.output_text))
            parts: list[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for block in getattr(item, "content", []) or []:
                        if getattr(block, "type", None) == "output_text":
                            parts.append(getattr(block, "text", "") or "")
            text = "\n".join(parts).strip()
            if text:
                return _strip_markdown(text)
        except Exception:
            continue
    return ""


# ---------------------------------------------------------------------------
# 차트 감지 & 렌더링
# ---------------------------------------------------------------------------
def detect_chart_indicator(question: str) -> str | None:
    trend_kw = ("추이", "흐름", "동향", "그래프", "차트", "변화", "추세", "경향", "움직")
    if not any(k in question for k in trend_kw):
        return None
    if any(k in question for k in ("환율", "달러", "원달러", "usd", "원/달러")):
        return "exchange"
    if any(k in question for k in ("금리", "기준금리")):
        return "rate"
    if any(k in question for k in ("물가", "cpi", "소비자물가", "인플레")):
        return "cpi"
    return None


def build_plotly_chart(indicator: str, df: pd.DataFrame, cycle: str) -> go.Figure | None:
    if df.empty:
        return None

    if indicator == "cpi":
        enriched = compute_cpi_yoy(df)
        plot_df = filter_display_period(enriched)
        if plot_df.empty:
            return None
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=plot_df["DATE"], y=plot_df["VALUE"],
                name="CPI 지수", mode="lines+markers",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["DATE"], y=plot_df["YoY"],
                name="전년동월비(%)", mode="lines+markers",
                line={"dash": "dot"},
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="소비자물가지수(CPI) 추이",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.12},
        )
        fig.update_yaxes(title_text="CPI 지수", secondary_y=False)
        fig.update_yaxes(title_text="YoY (%)", secondary_y=True)
        _apply_monthly_xaxis(fig)
        return fig

    plot_df = filter_display_period(df)
    if plot_df.empty:
        return None

    labels = {
        "exchange": ("원/달러 환율 추이", "환율 (원)"),
        "rate": ("한국은행 기준금리 추이", "금리 (%)"),
    }
    title, ylabel = labels.get(indicator, ("지표 추이", "값"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["DATE"], y=plot_df["VALUE"],
            mode="lines+markers", name=ylabel,
        )
    )
    fig.update_layout(title=title, yaxis_title=ylabel, hovermode="x unified")
    if cycle == "D":
        _apply_daily_xaxis(fig)
    else:
        _apply_monthly_xaxis(fig)
    return fig


def _apply_daily_xaxis(fig: go.Figure) -> None:
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        tickangle=-30,
        nticks=8,
    )


def _apply_monthly_xaxis(fig: go.Figure) -> None:
    fig.update_xaxes(tickformat="%Y-%m", dtick="M1", tickangle=-30)


def save_static_chart(
    indicator: str,
    df: pd.DataFrame,
    cycle: str,
) -> str | None:
    """Matplotlib/Seaborn으로 output/charts/{지표키}_{타임스탬프}.png 저장."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_key = INDICATOR_CONFIGS[indicator]["chart_key"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = CHARTS_DIR / f"{chart_key}_{ts}.png"

    try:
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_style("whitegrid")

        if indicator == "cpi":
            enriched = compute_cpi_yoy(df)
            plot_df = filter_display_period(enriched)
            if plot_df.empty:
                return None
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(plot_df["DATE"], plot_df["VALUE"], marker="o", label="CPI 지수")
            ax1.set_ylabel("CPI 지수")
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax2 = ax1.twinx()
            ax2.plot(plot_df["DATE"], plot_df["YoY"], marker="s", color="orange", label="YoY(%)")
            ax2.set_ylabel("YoY (%)")
            ax1.set_title("소비자물가지수(CPI) 추이")
            fig.autofmt_xdate(rotation=30)
        else:
            plot_df = filter_display_period(df)
            if plot_df.empty:
                return None
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=plot_df, x="DATE", y="VALUE", marker="o")
            titles = {
                "exchange": "원/달러 환율 추이",
                "rate": "한국은행 기준금리 추이",
            }
            plt.title(titles.get(indicator, "지표 추이"))
            ax = plt.gca()
            if cycle == "D":
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.xticks(rotation=30, ha="right")
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close("all")
        return str(filepath)
    except Exception:
        plt.close("all")
        return None


# ---------------------------------------------------------------------------
# 데이터 로드 (앱 전역)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="한국은행 ECOS 데이터 수집 중...")
def load_all_macro_data(api_key: str) -> dict[str, Any]:
    end_d = _end_date_for_cycle("D")
    end_m = _end_date_for_cycle("M")

    exchange_df, ex_cycle = fetch_indicator(
        api_key, INDICATOR_CONFIGS["exchange"],
        DISPLAY_START_D, end_d, DISPLAY_START_M, end_m,
    )
    rate_df, rate_cycle = fetch_indicator(
        api_key, INDICATOR_CONFIGS["rate"],
        DISPLAY_START_D, end_d, DISPLAY_START_M, end_m,
    )
    cpi_df, cpi_cycle = fetch_indicator(
        api_key, INDICATOR_CONFIGS["cpi"],
        DISPLAY_START_D, end_d, CPI_FETCH_START_M, end_m,
    )

    return {
        "exchange": exchange_df,
        "exchange_cycle": ex_cycle,
        "rate": rate_df,
        "rate_cycle": rate_cycle,
        "cpi": cpi_df,
        "cpi_cycle": cpi_cycle,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_kpi_row(data: dict[str, Any]) -> None:
    ex_kpi = kpi_exchange(data["exchange"])
    rt_kpi = kpi_rate(data["rate"])
    cp_kpi = kpi_cpi_yoy(data["cpi"])

    c1, c2, c3 = st.columns(3)

    with c1:
        if ex_kpi["value"] is not None:
            st.metric(
                "현재 원/달러 환율",
                f"{ex_kpi['value']:,.2f}원",
                delta=f"{ex_kpi['delta']:+.2f}원" if ex_kpi["delta"] is not None else None,
            )
            st.caption(format_as_of(ex_kpi["as_of"], "D"))
        else:
            st.metric("현재 원/달러 환율", "데이터 없음")

    with c2:
        if rt_kpi["value"] is not None:
            st.metric(
                "한국은행 기준금리",
                f"{rt_kpi['value']:.2f}%",
                delta=f"{rt_kpi['delta']:+.2f}%p" if rt_kpi["delta"] is not None else None,
            )
            st.caption(format_as_of(rt_kpi["as_of"], data["rate_cycle"]))
        else:
            st.metric("한국은행 기준금리", "데이터 없음")

    with c3:
        if cp_kpi["value"] is not None:
            st.metric(
                "소비자물가 전년동월비",
                f"{cp_kpi['value']:.1f}%",
                delta=f"{cp_kpi['delta']:+.2f}%p" if cp_kpi["delta"] is not None else None,
            )
            st.caption(format_as_of(cp_kpi["as_of"], "M"))
        else:
            st.metric("소비자물가 전년동월비", "데이터 없음")


def render_chat(data: dict[str, Any], client: OpenAI, model: str, use_web: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_indicator"):
                ind = msg["chart_indicator"]
                cycle_key = f"{ind}_cycle" if ind != "cpi" else "cpi_cycle"
                cycle = data.get(cycle_key, "M")
                fig = build_plotly_chart(ind, data[ind], cycle)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    prompt = st.chat_input("환율, 금리, 물가 등 거시경제 지표에 대해 질문하세요.")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = build_data_context(data["exchange"], data["rate"], data["cpi"])
    web_context = ""
    if use_web and needs_global_web_search(prompt):
        with st.spinner("글로벌 경제 트렌드 검색 중..."):
            web_context = run_web_search(client, prompt)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            answer = call_llm(client, model, prompt, context, web_context)
        st.markdown(answer)

        chart_indicator = detect_chart_indicator(prompt)
        chart_fig = None
        if chart_indicator:
            cycle_key = f"{chart_indicator}_cycle" if chart_indicator != "cpi" else "cpi_cycle"
            cycle = data.get(cycle_key, "M")
            chart_fig = build_plotly_chart(chart_indicator, data[chart_indicator], cycle)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
                save_static_chart(chart_indicator, data[chart_indicator], cycle)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chart_indicator": chart_indicator if chart_fig else None,
    })


def main() -> None:
    st.set_page_config(
        page_title="거시경제 지표 AI 분석",
        page_icon="📈",
        layout="wide",
    )

    merged_env, checked_paths = load_merged_env()
    ecos_key, openai_key, missing = resolve_api_keys(merged_env)

    st.title("📈 API 기반 환율 및 거시경제 지표 AI 분석 에이전트")
    st.caption("데이터 구간: 2026년 1월 ~ 현재 (CPI 전년동월비 산출 시 2025년 동월치 참조)")

    with st.sidebar:
        st.header("제어 패널")
        model = st.selectbox(
            "LLM 모델",
            ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
            index=0,
        )
        use_web = st.checkbox("OpenAI 실시간 글로벌 금융 트렌드 웹 검색 (web_search)", value=False)
        if st.button("대화 초기화 (Reset Chat)", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if missing:
        show_key_error(missing, checked_paths)
        return

    client = OpenAI(api_key=openai_key)

    try:
        data = load_all_macro_data(ecos_key)
    except Exception as exc:
        st.error(f"ECOS API 데이터 수집 중 오류가 발생했습니다: {exc}")
        return

    render_kpi_row(data)
    st.divider()
    render_chat(data, client, model, use_web)


if __name__ == "__main__":
    main()
