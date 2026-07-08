-- multiusers-ref.sql
-- 멀티유저 RAG 챗봇(multiusers.py)용 Supabase 스키마
-- Supabase SQL Editor에서 처음부터 끝까지 한 번에 실행하세요.

-- ---------------------------------------------------------------------------
-- 기존 객체 정리
-- ---------------------------------------------------------------------------
DROP FUNCTION IF EXISTS match_vector_documents(vector, int, uuid, bigint);
DROP FUNCTION IF EXISTS match_vector_documents(vector, int, uuid);
DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS vector_documents CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;

-- ---------------------------------------------------------------------------
-- 확장
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 사용자 (Supabase Auth 미사용 — 앱에서 login_id/password_hash 관리)
-- ---------------------------------------------------------------------------
CREATE TABLE "user" (
    id              BIGSERIAL PRIMARY KEY,
    login_id        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_login_id ON "user" (login_id);

-- ---------------------------------------------------------------------------
-- 채팅 세션 / 메시지 / 벡터 문서
-- ---------------------------------------------------------------------------
CREATE TABLE chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     BIGINT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '새 세션',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE vector_documents (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    file_name   TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_sessions_user_id ON chat_sessions (user_id);
CREATE INDEX idx_chat_messages_user_session ON chat_messages (user_id, session_id);
CREATE INDEX idx_vector_documents_user_session ON vector_documents (user_id, session_id);
CREATE INDEX idx_vector_documents_file_name ON vector_documents (user_id, session_id, file_name);

-- ---------------------------------------------------------------------------
-- 벡터 유사도 검색 RPC (user_id + session_id 이중 필터)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_vector_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    filter_session_id uuid DEFAULT NULL,
    filter_user_id bigint DEFAULT NULL
)
RETURNS TABLE (
    id          bigint,
    session_id  uuid,
    file_name   text,
    content     text,
    metadata    jsonb,
    similarity  float
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        vd.id,
        vd.session_id,
        vd.file_name,
        vd.content,
        vd.metadata,
        1 - (vd.embedding <=> query_embedding) AS similarity
    FROM vector_documents vd
    WHERE vd.embedding IS NOT NULL
      AND (filter_session_id IS NULL OR vd.session_id = filter_session_id)
      AND (filter_user_id IS NULL OR vd.user_id = filter_user_id)
    ORDER BY vd.embedding <=> query_embedding
    LIMIT GREATEST(match_count, 1);
$$;

-- ---------------------------------------------------------------------------
-- updated_at 자동 갱신
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_chat_sessions_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW
    EXECUTE FUNCTION set_chat_sessions_updated_at();

-- ---------------------------------------------------------------------------
-- RLS (anon 키로 앱에서 CRUD 가능하도록)
-- ---------------------------------------------------------------------------
ALTER TABLE "user" ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE vector_documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_all" ON "user"
    FOR ALL TO anon, authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "chat_sessions_all" ON chat_sessions
    FOR ALL TO anon, authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "chat_messages_all" ON chat_messages
    FOR ALL TO anon, authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "vector_documents_all" ON vector_documents
    FOR ALL TO anon, authenticated
    USING (true)
    WITH CHECK (true);
