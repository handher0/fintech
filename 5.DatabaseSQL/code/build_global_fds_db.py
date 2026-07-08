"""글로벌 결제 CSV → 로컬 SQLite(global_fds.db) + Supabase 업로드 빌드 스크립트."""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "global_payments.csv"
SQLITE_PATH = DATA_DIR / "global_fds.db"

# ---------------------------------------------------------------------------
# Supabase 연결 정보 (플레이스홀더 — .env 또는 환경 변수로 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
YOUR_SUPABASE_URL = "YOUR_SUPABASE_URL"
YOUR_SUPABASE_KEY = "YOUR_SUPABASE_KEY"

TABLE_NAME = "global_payments"
BATCH_SIZE = 200

COLUMN_MAP = {
    "Transaction_ID": "transaction_id",
    "User_ID": "user_id",
    "Timestamp": "timestamp",
    "Amount_USD": "amount_usd",
    "Country": "country",
    "Merchant_Category": "merchant_category",
    "Device_IP": "device_ip",
    "Is_Fraud": "is_fraud",
}

DB_COLUMNS = list(COLUMN_MAP.values())


def load_supabase_credentials() -> tuple[str, str]:
    """Supabase URL·API Key를 환경 변수 또는 플레이스홀더에서 읽는다.

    벌크 업로드 시 SUPABASE_SERVICE_ROLE_KEY가 있으면 우선 사용한다.
    (service_role은 RLS를 우회하므로 서버 사이드 적재에 적합)
    없으면 SUPABASE_ANON_KEY를 사용하며, 이 경우 RLS 정책이 필요하다.
    """
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)

    url = os.getenv("SUPABASE_URL", YOUR_SUPABASE_URL).strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if key:
        return url, key

    key = os.getenv("SUPABASE_ANON_KEY", os.getenv("SUPABASE_KEY", YOUR_SUPABASE_KEY)).strip()
    return url, key


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """CSV를 로드하고 기본 검증을 수행한다."""
    print(f"[1/5] CSV 로드: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  - 원본 행 수: {len(df):,}")
    print(f"  - 컬럼: {list(df.columns)}")

    missing_cols = [c for c in COLUMN_MAP if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    # 컬럼명을 DB 스키마(snake_case)로 통일
    df = df.rename(columns=COLUMN_MAP)
    print("  - 컬럼명 변환 완료 (snake_case)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Amount_USD가 0 이하이거나 null인 행을 삭제한다."""
    print("[2/5] 데이터 정제 (비정상 결제 금액 제거)")
    before = len(df)

    amount = pd.to_numeric(df["amount_usd"], errors="coerce")
    invalid_mask = amount.isna() | (amount <= 0)
    removed = int(invalid_mask.sum())

    cleaned = df.loc[~invalid_mask].copy()
    cleaned["amount_usd"] = amount[~invalid_mask]
    cleaned["is_fraud"] = pd.to_numeric(cleaned["is_fraud"], errors="coerce").fillna(0).astype(int)

    print(f"  - 제거된 행: {removed:,}")
    print(f"  - 정제 후 행 수: {len(cleaned):,} (원본 {before:,} → {len(cleaned):,})")
    return cleaned.reset_index(drop=True)


def sanitize_value(value):
    """NaN·Inf를 None으로 치환해 JSON/DB 업로드 오류를 방지한다."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def records_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """DataFrame을 Supabase/SQLite 공통 dict 레코드 리스트로 변환한다."""
    records: list[dict] = []
    for row in df[DB_COLUMNS].to_dict(orient="records"):
        record = {col: sanitize_value(val) for col, val in row.items()}
        records.append(record)
    return records


def build_sqlite_db(records: list[dict], db_path: Path) -> None:
    """로컬 SQLite DB를 생성하고 데이터를 insert 한다."""
    print(f"[3/5] SQLite DB 생성: {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print("  - 기존 DB 파일 삭제 후 재생성")
        db_path.unlink()

    create_sql = f"""
    CREATE TABLE {TABLE_NAME} (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT,
        timestamp TEXT,
        amount_usd REAL,
        country TEXT,
        merchant_category TEXT,
        device_ip TEXT,
        is_fraud INTEGER
    );
    """

    index_sql = [
        f"CREATE INDEX idx_{TABLE_NAME}_user_id ON {TABLE_NAME}(user_id);",
        f"CREATE INDEX idx_{TABLE_NAME}_timestamp ON {TABLE_NAME}(timestamp);",
        f"CREATE INDEX idx_{TABLE_NAME}_country ON {TABLE_NAME}(country);",
        f"CREATE INDEX idx_{TABLE_NAME}_is_fraud ON {TABLE_NAME}(is_fraud);",
    ]

    insert_sql = f"""
    INSERT INTO {TABLE_NAME} (
        transaction_id, user_id, timestamp, amount_usd,
        country, merchant_category, device_ip, is_fraud
    ) VALUES (
        :transaction_id, :user_id, :timestamp, :amount_usd,
        :country, :merchant_category, :device_ip, :is_fraud
    );
    """

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(create_sql)
            for stmt in index_sql:
                conn.execute(stmt)
            conn.executemany(insert_sql, records)
            conn.commit()

            count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        print(f"  - SQLite insert 완료: {count:,}건")
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite 저장 중 오류: {exc}") from exc


def upload_to_supabase(records: list[dict], url: str, key: str) -> None:
    """Supabase에 배치(200건) 단위로 데이터를 업로드한다."""
    print("[4/5] Supabase 업로드 준비")

    if url == "YOUR_SUPABASE_URL" or key == "YOUR_SUPABASE_KEY":
        print("  - Supabase URL/Key가 설정되지 않아 업로드를 건너뜁니다.")
        print("    → YOUR_SUPABASE_URL / YOUR_SUPABASE_KEY 를 수정하거나")
        print("    → .env 에 SUPABASE_URL, SUPABASE_ANON_KEY 를 설정하세요.")
        return

    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError(
            "supabase-py 패키지가 필요합니다. `pip install supabase` 로 설치하세요."
        ) from exc

    print(f"  - 대상 URL: {url}")
    print(f"  - 총 레코드: {len(records):,}건, 배치 크기: {BATCH_SIZE}")

    try:
        client = create_client(url, key)
        total_batches = math.ceil(len(records) / BATCH_SIZE)

        for i in range(0, len(records), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = records[i : i + BATCH_SIZE]
            print(f"  - 배치 {batch_num}/{total_batches} 업로드 중 ({len(batch)}건)...")

            # upsert: transaction_id 기준 중복 시 갱신
            client.table(TABLE_NAME).upsert(batch, on_conflict="transaction_id").execute()

        print("[5/5] Supabase 업로드 완료")
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if "global_payments" in err_msg and ("PGRST205" in err_msg or "schema cache" in err_msg):
            print("  - [안내] Supabase에 global_payments 테이블이 없습니다.")
            print("    → code/create_supabase_table.sql 내용을 Supabase SQL Editor에서 먼저 실행하세요.")
            return
        if "row-level security" in err_msg.lower() or "42501" in err_msg:
            print("  - [안내] RLS(행 수준 보안) 정책 때문에 INSERT가 거부되었습니다.")
            print("    → Supabase SQL Editor에서 create_supabase_table.sql 의 RLS 정책 부분을 실행하거나")
            print("    → .env 에 SUPABASE_SERVICE_ROLE_KEY 를 추가한 뒤 다시 실행하세요.")
            return
        raise RuntimeError(f"Supabase 업로드 중 오류: {exc}") from exc


def main() -> None:
    """CSV 정제 → SQLite 저장 → Supabase 업로드 전체 파이프라인."""
    print("=" * 60)
    print("글로벌 결제 DB 빌드 시작")
    print("=" * 60)

    try:
        df = load_and_validate_csv(CSV_PATH)
        cleaned = clean_data(df)
        records = records_from_dataframe(cleaned)

        build_sqlite_db(records, SQLITE_PATH)

        supabase_url, supabase_key = load_supabase_credentials()
        upload_to_supabase(records, supabase_url, supabase_key)

        print("=" * 60)
        print("모든 작업이 완료되었습니다.")
        print(f"  - SQLite: {SQLITE_PATH}")
        print(f"  - 정제된 레코드 수: {len(records):,}")
        print("=" * 60)
    except Exception as exc:  # noqa: BLE001
        print("=" * 60)
        print(f"[오류] 작업 실패: {exc}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
