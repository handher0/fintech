-- global_payments 테이블 생성 스크립트
-- Supabase SQL Editor에 복사·붙여넣기 후 실행하세요.

DROP TABLE IF EXISTS global_payments;

CREATE TABLE global_payments (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT,
    timestamp TIMESTAMP,
    amount_usd NUMERIC,
    country TEXT,
    merchant_category TEXT,
    device_ip TEXT,
    is_fraud INTEGER
);

CREATE INDEX idx_global_payments_user_id ON global_payments (user_id);
CREATE INDEX idx_global_payments_timestamp ON global_payments (timestamp);
CREATE INDEX idx_global_payments_country ON global_payments (country);
CREATE INDEX idx_global_payments_is_fraud ON global_payments (is_fraud);

-- RLS(Row Level Security): Supabase public 테이블은 기본적으로 RLS가 켜져 있어
-- 정책 없이는 anon 키로 INSERT/UPDATE가 거부됩니다 (오류 42501).
ALTER TABLE global_payments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "global_payments_select" ON global_payments
    FOR SELECT TO anon, authenticated
    USING (true);

CREATE POLICY "global_payments_insert" ON global_payments
    FOR INSERT TO anon, authenticated
    WITH CHECK (true);

CREATE POLICY "global_payments_update" ON global_payments
    FOR UPDATE TO anon, authenticated
    USING (true)
    WITH CHECK (true);
