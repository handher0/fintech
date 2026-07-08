"""
스마트 경제 뉴스 메타 태깅 플랫폼

경제 뉴스·리포트 원문 또는 URL에서 본문을 추출하고,
한국은행 ECOS 통계용어사전과 연동해 용어 툴팁 태깅 및 AI 브리핑을 제공한다.

실행 방법:
  pip install streamlit python-dotenv openai requests beautifulsoup4
  cd 7.BoKMacro
  streamlit run code/macro_news_tagger.py
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
SEED_PATH = DATA_DIR / "seed_glossary.json"
CACHE_PATH = DATA_DIR / "glossary_cache.json"

ECOS_WORD_URL = "http://ecos.bok.or.kr/api/StatisticWord"

SEARCH_PREFIXES: list[str] = (
    ["가", "까", "나", "다", "따", "라", "마", "바", "빠", "사", "싸", "아", "자", "짜", "차", "카", "타", "파", "하"]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
)

ARTICLE_SELECTORS = [
    "article",
    "main",
    "#dic_area",
    ".news_body",
    "#articeBody",
    "#articleBodyContents",
    ".article_body",
    ".news_view",
    ".entry-content",
    "#newsct_article",
]

BRIEFING_SYSTEM_PROMPT = """너는 글로벌 금융시장을 분석하는 수석 매크로 이코노미스트다.

답변 형식 규칙 (반드시 준수):
- 마크다운 기호(#, ##, **, `, >, ---)를 절대 사용하지 마세요.
- 일반 문단 텍스트와 불렛 리스트(-)만 사용하세요.
- 아래 두 섹션을 순서대로 작성하세요.

1) 핵심 요약 3줄 (각 줄 한 문장, - 로 시작)
2) 거시경제 지표가 주식·채권·환율·원자재 시장에 미칠 영향 분석 (불렛 3~5개)
"""


class EcosRateLimitError(Exception):
    """ECOS API 호출 한도 초과 (ERROR-602)."""


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


def load_merged_env() -> tuple[dict[str, str], list[Path]]:
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
        "API Key가 로드되지 않았습니다. .env 파일의 변수명(ECOS_API_KEY, OPENAI_API_KEY)과 "
        "파일 경로를 재확인해 주세요."
    )
    st.caption(f"누락된 변수: {', '.join(missing)}")
    if checked_paths:
        st.caption("확인한 .env 경로: " + " | ".join(str(p) for p in checked_paths))
    else:
        st.caption(f"확인한 .env 경로: {BASE_DIR / '.env'} | {PROJECT_ROOT / '.env'} (파일 없음)")


# ---------------------------------------------------------------------------
# 용어 사전
# ---------------------------------------------------------------------------
def load_seed_glossary() -> dict[str, str]:
    if not SEED_PATH.exists():
        return {}
    try:
        data = json.loads(SEED_PATH.read_text(encoding="utf-8-sig"))
        return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_statistic_word_rows(payload: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    if "RESULT" in payload:
        code = str(payload["RESULT"].get("CODE", ""))
        if code == "INFO-200":
            return [], 0
        if code == "ERROR-602":
            raise EcosRateLimitError(payload["RESULT"].get("MESSAGE", "호출 한도 초과"))
        if code and code not in ("INFO-000", ""):
            msg = payload["RESULT"].get("MESSAGE", code)
            raise RuntimeError(f"ECOS StatisticWord 오류: {code} — {msg}")

    block = payload.get("StatisticWord") or {}
    total = int(block.get("list_total_count") or 0)
    rows = block.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    return rows, total


def fetch_glossary_from_ecos(api_key: str) -> dict[str, str]:
    """접두어별 페이지네이션으로 ECOS 통계용어사전 전체를 수집한다."""
    glossary: dict[str, str] = {}
    page_size = 1000

    for prefix in SEARCH_PREFIXES:
        start = 1
        while True:
            end = start + page_size - 1
            url = f"{ECOS_WORD_URL}/{api_key}/json/kr/{start}/{end}/{quote(prefix, safe='')}"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except EcosRateLimitError:
                raise
            except Exception as exc:
                raise RuntimeError(f"ECOS API 통신 실패 (접두어={prefix}): {exc}") from exc

            rows, total_count = _parse_statistic_word_rows(payload)
            for row in rows:
                word = str(row.get("WORD", "")).strip()
                content = str(row.get("CONTENT", "")).strip()
                if word and content:
                    glossary[word] = content

            if not rows or end >= total_count or total_count == 0:
                break
            start += page_size
            time.sleep(0.05)

        time.sleep(0.05)

    return glossary


def _merge_glossaries(*sources: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for src in sources:
        merged.update(src)
    return merged


@st.cache_resource(show_spinner="경제 용어 사전 로드 중...")
def load_glossary_dictionary(api_key: str) -> tuple[dict[str, str], str]:
    """3단계 우선순위: 캐시 → API → 시드 폴백."""
    seed = load_seed_glossary()

    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8-sig"))
            cache_dict = {str(k): str(v) for k, v in cached.items()}
            return _merge_glossaries(seed, cache_dict), "cache"
        except (OSError, json.JSONDecodeError):
            pass

    try:
        api_dict = fetch_glossary_from_ecos(api_key)
        if api_dict:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps(api_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return _merge_glossaries(seed, api_dict), "api"
        st.warning("ECOS API에서 수집된 용어가 없어 시드 사전으로 폴백합니다.")
        return seed, "rate_limit_fallback"
    except EcosRateLimitError:
        st.warning("ECOS API 호출 한도(ERROR-602)에 도달하여 시드 사전으로 폴백합니다.")
        return seed, "rate_limit_fallback"
    except Exception as exc:
        st.warning(f"ECOS 용어 사전 수집 실패: {exc} — 시드 사전으로 폴백합니다.")
        return seed, "rate_limit_fallback"


# ---------------------------------------------------------------------------
# 태깅 · HTML
# ---------------------------------------------------------------------------
def find_terms_in_text(text: str, glossary: dict[str, str]) -> tuple[list[tuple[int, int, str]], list[str]]:
    """본문에서 사전 용어를 찾아 (start, end, term) 구간과 고유 용어 목록을 반환한다."""
    if not text or not glossary:
        return [], []

    terms = sorted(
        (t for t in glossary if len(t) >= 2),
        key=len,
        reverse=True,
    )
    occupied: list[tuple[int, int]] = []
    matches: list[tuple[int, int, str]] = []

    for term in terms:
        for m in re.finditer(re.escape(term), text):
            s, e = m.start(), m.end()
            if any(not (e <= os_s or s >= os_e) for os_s, os_e in occupied):
                continue
            occupied.append((s, e))
            matches.append((s, e, term))

    matches.sort(key=lambda x: x[0])
    unique = list(dict.fromkeys(m[2] for m in matches))
    return matches, unique


def build_tagged_html(text: str, glossary: dict[str, str]) -> tuple[str, list[str]]:
    matches, unique_terms = find_terms_in_text(text, glossary)
    if not matches:
        return f"<p>{html.escape(text)}</p>", []

    parts: list[str] = []
    cursor = 0
    for s, e, term in matches:
        if cursor < s:
            parts.append(html.escape(text[cursor:s]))
        definition = html.escape(glossary.get(term, ""), quote=True)
        parts.append(
            f'<span class="eco-term" tabindex="0" data-definition="{definition}">'
            f"{html.escape(term)}</span>"
        )
        cursor = e
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return f'<p style="white-space:pre-wrap;">{"".join(parts)}</p>', unique_terms


def _estimate_render_height(text: str) -> int:
    lines = max(1, text.count("\n") + 1, len(text) // 60)
    return min(900, max(280, lines * 26 + 80))


def render_tagged_news(tagged_html: str, text: str) -> None:
    height = _estimate_render_height(text)
    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
    font-size: 16px;
    line-height: 1.75;
    color: #1a1a1a;
    margin: 0;
    padding: 8px 4px 24px;
  }}
  .eco-term {{
    color: #1565c0;
    border-bottom: 1px dashed #1565c0;
    cursor: help;
    background: rgba(21, 101, 192, 0.06);
    padding: 0 2px;
    border-radius: 2px;
  }}
  #eco-float-tip {{
    display: none;
    position: fixed;
    z-index: 99999;
    max-width: 360px;
    background: #1e293b;
    color: #f8fafc;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    pointer-events: none;
    word-break: keep-all;
  }}
  #eco-float-tip::after {{
    content: "";
    position: absolute;
    left: var(--arrow-left, 20px);
    border: 7px solid transparent;
  }}
  #eco-float-tip.tip-below::after {{
    top: -14px;
    border-bottom-color: #1e293b;
  }}
  #eco-float-tip.tip-above::after {{
    bottom: -14px;
    border-top-color: #1e293b;
  }}
</style>
</head>
<body>
<div id="content">{tagged_html}</div>
<div id="eco-float-tip"></div>
<script>
(function() {{
  const tip = document.getElementById('eco-float-tip');
  const margin = 12;

  function hideTip() {{
    tip.style.display = 'none';
  }}

  function showTip(el) {{
    const def = el.getAttribute('data-definition');
    if (!def) return;
    tip.textContent = def;
    tip.style.display = 'block';
    tip.classList.remove('tip-above', 'tip-below');

    const rect = el.getBoundingClientRect();
    const tipRect = tip.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - tipRect.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - tipRect.width - margin));

    let top = rect.bottom + 10;
    let above = false;
    if (top + tipRect.height > window.innerHeight - margin) {{
      top = rect.top - tipRect.height - 10;
      above = true;
    }}
    top = Math.max(margin, Math.min(top, window.innerHeight - tipRect.height - margin));

    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
    tip.classList.add(above ? 'tip-above' : 'tip-below');

    const arrowLeft = Math.max(12, Math.min(rect.left + rect.width / 2 - left, tipRect.width - 20));
    tip.style.setProperty('--arrow-left', arrowLeft + 'px');
  }}

  document.querySelectorAll('.eco-term').forEach(el => {{
    el.addEventListener('mouseenter', () => showTip(el));
    el.addEventListener('focus', () => showTip(el));
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('blur', hideTip);
  }});
}})();
</script>
</body>
</html>
        """,
        height=height,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# URL 본문 추출
# ---------------------------------------------------------------------------
def is_valid_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def extract_article_from_url(url: str) -> tuple[str, str]:
    """URL에서 (제목, 본문)을 추출한다."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url.strip(), headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()

    container = None
    for sel in ARTICLE_SELECTORS:
        container = soup.select_one(sel)
        if container:
            break
    if container is None:
        container = soup.body or soup

    paragraphs: list[str] = []
    for tag in container.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = tag.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if len(text) >= 20:
            paragraphs.append(text)

    body = "\n\n".join(paragraphs)
    if len(body) < 50:
        raise ValueError("본문을 충분히 추출하지 못했습니다. 원문 직접 입력 탭을 이용해 주세요.")
    return title, body


# ---------------------------------------------------------------------------
# AI 브리핑
# ---------------------------------------------------------------------------
def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def call_ai_briefing(
    client: OpenAI,
    model: str,
    article_text: str,
    matched_terms: list[str],
) -> str:
    terms_block = ", ".join(matched_terms[:30]) if matched_terms else "(식별된 용어 없음)"
    user_msg = (
        f"[기사 본문]\n{article_text[:6000]}\n\n"
        f"[식별된 경제 용어]\n{terms_block}\n\n"
        "위 기사를 분석하여 핵심 요약 3줄과 시장 영향 분석을 작성하세요."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BRIEFING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return _strip_markdown((resp.choices[0].message.content or "").strip())
    except Exception as exc:
        return f"AI 브리핑 생성 중 오류가 발생했습니다: {exc}"


# ---------------------------------------------------------------------------
# 분석 파이프라인
# ---------------------------------------------------------------------------
def run_news_analysis(
    text: str,
    source: str,
    glossary: dict[str, str],
    client: OpenAI,
    model: str,
) -> None:
    tagged_html, matched = build_tagged_html(text, glossary)
    briefing = call_ai_briefing(client, model, text, matched)

    st.session_state.news_text = text
    st.session_state.tagged_html = tagged_html
    st.session_state.matched_terms = matched
    st.session_state.ai_briefing = briefing
    st.session_state.article_source = source


# ---------------------------------------------------------------------------
# 세션 · UI
# ---------------------------------------------------------------------------
def init_session() -> None:
    defaults = {
        "news_text": "",
        "tagged_html": "",
        "matched_terms": [],
        "ai_briefing": "",
        "article_source": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_screen() -> None:
    st.session_state.news_text = ""
    st.session_state.tagged_html = ""
    st.session_state.matched_terms = []
    st.session_state.ai_briefing = ""
    st.session_state.article_source = ""


def _load_source_label(source: str) -> str:
    labels = {
        "cache": "로컬 캐시 + 시드 사전",
        "api": "ECOS API 동기화 + 시드 사전",
        "rate_limit_fallback": "시드 사전 폴백 (API 한도·오류)",
    }
    return labels.get(source, source)


def main() -> None:
    st.set_page_config(page_title="경제 뉴스 메타 태깅", page_icon="📰", layout="wide")
    init_session()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    merged_env, checked_paths = load_merged_env()
    ecos_key, openai_key, missing = resolve_api_keys(merged_env)

    st.title("스마트 경제 뉴스 메타 태깅 플랫폼")
    st.caption("경제 전문 용어를 자동 식별하고 마우스 오버 시 한국은행 통계용어 설명을 제공합니다.")

    with st.sidebar:
        st.header("제어 패널")
        model = st.selectbox("LLM 모델", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0)
        if st.button("화면 초기화 (Reset)", use_container_width=True):
            reset_screen()
            st.rerun()
        if CACHE_PATH.exists():
            st.caption(f"로컬 캐시: {CACHE_PATH.name} 사용 가능")

    if missing:
        show_key_error(missing, checked_paths)
        return

    try:
        client = OpenAI(api_key=openai_key)
        glossary, load_source = load_glossary_dictionary(ecos_key)
    except Exception as exc:
        st.error(f"용어 사전 로드 실패: {exc}")
        return

    st.info(f"로드된 경제 용어: {len(glossary):,}개 · 출처: {_load_source_label(load_source)}")

    tab_text, tab_url = st.tabs(["원문 직접 입력", "기사 링크 추출"])

    with tab_text:
        raw = st.text_area("경제 뉴스 · 리포트 원문", height=220, placeholder="기사 또는 리포트 본문을 붙여넣으세요.")
        if st.button("변환 분석", use_container_width=True, type="primary"):
            if not raw or not raw.strip():
                st.warning("분석할 본문을 입력해 주세요.")
            else:
                with st.spinner("용어 태깅 및 AI 브리핑 생성 중..."):
                    try:
                        run_news_analysis(raw.strip(), "원문 직접 입력", glossary, client, model)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"분석 중 오류: {exc}")

    with tab_url:
        url = st.text_input("기사 링크", placeholder="https://...")
        if st.button("본문 추출 및 분석", use_container_width=True, type="primary"):
            if not is_valid_url(url):
                st.error("http 또는 https로 시작하는 유효한 URL을 입력해 주세요.")
            else:
                with st.spinner("본문 추출 및 분석 중..."):
                    try:
                        title, body = extract_article_from_url(url)
                        with st.expander("추출된 제목·원문 미리보기"):
                            st.markdown(f"**제목:** {title}")
                            st.text(body[:2000] + ("..." if len(body) > 2000 else ""))
                        run_news_analysis(body, f"URL: {url.strip()}", glossary, client, model)
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"네트워크 오류로 본문을 가져오지 못했습니다: {exc}\n원문 직접 입력 탭을 이용해 주세요.")
                    except Exception as exc:
                        st.error(f"{exc}\n원문 직접 입력 탭을 이용해 주세요.")

    if st.session_state.tagged_html:
        st.divider()
        st.subheader("용어 태깅 결과")
        st.caption(f"분석 출처: {st.session_state.article_source}")
        matched = st.session_state.matched_terms
        st.caption(f"식별 용어 {len(matched)}개: {', '.join(matched) if matched else '없음'}")
        render_tagged_news(st.session_state.tagged_html, st.session_state.news_text)

        st.subheader("AI 에이전트 요약 코너")
        st.text(st.session_state.ai_briefing)


if __name__ == "__main__":
    main()
