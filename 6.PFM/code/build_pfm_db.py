"""
가계부 통합 CSV → Supabase(user_card_history) 배치 업로드 스크립트

전제:
  Supabase에 user_card_history 테이블이 이미 생성되어 있어야 함
  (code/create_table.sql 을 Supabase SQL Editor에서 먼저 실행)

기능:
  CSV 로드 → Total 행/결측치 정제 → NaN/Inf → None → 1,000개 단위 Bulk Insert

실행 방법:
  1. 패키지 설치: pip install supabase pandas
  2. 실행 명령어: python code/build_pfm_db.py
"""

import math
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    from supabase import create_client
except ModuleNotFoundError:
    print("[오류] 필요 라이브러리가 없습니다. 'pip install supabase pandas'를 실행해주세요.")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


# ---------------------------------------------------------------------------
# 경로 & 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "user_card_history_merged.csv"

TABLE_NAME = "user_card_history"
CHUNK_SIZE = 1000

# ---------------------------------------------------------------------------
# Supabase 연결 정보 (환경 변수 또는 직접 입력)
# ---------------------------------------------------------------------------
SUPABASE_URL = ""  # 예: "https://xxxx.supabase.co" (비워두면 .env에서 로드)
SUPABASE_KEY = ""  # 예: "eyJhbGci..." (비워두면 .env에서 로드)

# CSV 원본 컬럼 → DB 컬럼(snake_case) 매핑
COLUMN_MAP = {
    "month": "month",
    "Approved_Num": "approved_num",
    "Approved_DateTime": "approved_date_time",
    "Card_Name": "card_name",
    "Store_Name": "store_name",
    "Category": "category",
    "Amount_KRW": "amount_krw",
}


def resolve_credentials() -> tuple[str, str]:
    """Supabase URL/Key를 상수 또는 .env 환경 변수에서 확보한다."""
    url, key = SUPABASE_URL.strip(), SUPABASE_KEY.strip()

    if (not url or not key) and load_dotenv is not None:
        repo_root = BASE_DIR.parent
        load_dotenv(repo_root / ".env", override=False)

    if not url:
        url = os.getenv("SUPABASE_URL", "").strip()
    if not key:
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_ANON_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
        )
    return url, key


def load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """CSV를 로드하고 Total 행/결측치를 정제한다."""
    print(f"[1단계] CSV 로드: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  - 원본 행 수: {len(df):,}")

    before = len(df)

    # month 컬럼이 'Total'인 잔재 행 제거
    if "month" in df.columns:
        df = df[df["month"].astype(str).str.strip().str.lower() != "total"]

    # Amount_KRW 정제: 숫자 변환 후 0/Null/빈 값 제외
    df["Amount_KRW"] = pd.to_numeric(df["Amount_KRW"], errors="coerce")
    df = df[df["Amount_KRW"].notna()]
    df = df[df["Amount_KRW"] > 0]

    print(f"  - 정제 후 행 수: {len(df):,} (제외 {before - len(df):,}건)")
    return df.reset_index(drop=True)


def sanitize(value):
    """NaN/Inf/빈 문자열을 None으로 치환한다."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def build_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame을 Supabase insert용 dict 리스트로 변환한다."""
    renamed = df.rename(columns=COLUMN_MAP)
    db_cols = list(COLUMN_MAP.values())
    records = []
    for row in renamed[db_cols].to_dict(orient="records"):
        clean = {col: sanitize(val) for col, val in row.items()}
        # amount_krw는 float 보장
        if clean.get("amount_krw") is not None:
            clean["amount_krw"] = float(clean["amount_krw"])
        records.append(clean)
    return records


def upload_in_batches(client, records: list[dict]) -> None:
    """1,000개 단위 청크로 Bulk Insert 한다."""
    total = len(records)
    total_chunks = math.ceil(total / CHUNK_SIZE)
    print(f"[3단계] Supabase 업로드 시작 (총 {total:,}건, 청크 {CHUNK_SIZE}개 단위)")

    for i in range(0, total, CHUNK_SIZE):
        chunk_num = i // CHUNK_SIZE + 1
        chunk = records[i : i + CHUNK_SIZE]
        print(f"  - 청크 {chunk_num}/{total_chunks} 업로드 중 ({len(chunk)}건)...")
        client.table(TABLE_NAME).insert(chunk).execute()

    print("  - 모든 청크 업로드 완료")


def main() -> None:
    """CSV 정제 → 레코드 빌드 → Supabase 배치 업로드."""
    print("=" * 60)
    print("가계부 데이터 Supabase 업로드 시작")
    print("=" * 60)

    try:
        url, key = resolve_credentials()
        if not url or not key:
            print("[오류] Supabase 접속 정보가 없습니다.")
            print("       코드 상단 SUPABASE_URL/SUPABASE_KEY를 채우거나")
            print("       .env에 SUPABASE_URL, SUPABASE_ANON_KEY(또는 SERVICE_ROLE_KEY)를 설정하세요.")
            sys.exit(1)

        df = load_and_clean_csv(CSV_PATH)
        print("[2단계] 업로드 레코드 빌드 및 결측치(None) 정제")
        records = build_records(df)
        print(f"  - 빌드된 레코드: {len(records):,}건")

        client = create_client(url, key)
        upload_in_batches(client, records)

        print("=" * 60)
        print(f"업로드 완료: {len(records):,}건")
        print("=" * 60)
    except Exception as exc:  # noqa: BLE001
        print("=" * 60)
        print(f"[오류] 업로드 실패: {exc}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
