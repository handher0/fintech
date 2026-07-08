-- user_card_history 테이블 생성 스크립트
-- Supabase SQL Editor에 복사·붙여넣기 후 실행하세요.

DROP TABLE IF EXISTS user_card_history;

CREATE TABLE user_card_history (
    month TEXT,
    approved_num TEXT,
    approved_date_time TEXT,
    card_name TEXT,
    store_name TEXT,
    category TEXT,
    amount_krw REAL
);

CREATE INDEX idx_user_card_history_month ON user_card_history (month);
CREATE INDEX idx_user_card_history_category ON user_card_history (category);
CREATE INDEX idx_user_card_history_card_name ON user_card_history (card_name);
CREATE INDEX idx_user_card_history_store_name ON user_card_history (store_name);

-- RLS(Row Level Security): anon 키로 INSERT하려면 정책이 필요합니다.
ALTER TABLE user_card_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_card_history_select" ON user_card_history
    FOR SELECT TO anon, authenticated
    USING (true);

CREATE POLICY "user_card_history_insert" ON user_card_history
    FOR INSERT TO anon, authenticated
    WITH CHECK (true);
