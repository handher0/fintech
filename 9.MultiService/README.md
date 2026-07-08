# 9. MultiService — 멀티세션·멀티유저 RAG 챗봇

Supabase와 OpenAI를 활용한 **PDF 기반 RAG 챗봇**입니다.  
대화 세션 저장·로드와 사용자별 데이터 분리를 지원합니다.

## 라이브 데모

| 앱 | 배포 URL |
|---|---|
| **멀티유저 RAG 챗봇** (`multiusers.py`) | https://handher0-fintech.streamlit.app/ |

회원가입 후 PDF 업로드 → 질문 → 세션 저장·로드까지 사용할 수 있습니다.

---

## 프로젝트 구성

```
9.MultiService/
├── code/
│   ├── multiref.py              # 멀티세션 RAG (단일 사용자)
│   ├── multiusers.py            # 멀티유저 RAG (로그인/회원가입)
│   ├── multi-session-ref.sql    # 멀티세션용 DB 스키마
│   ├── multiusers-ref.sql       # 멀티유저용 DB 스키마
│   └── requirements.txt
├── prompts/
│   ├── 멀티세션 ref.txt
│   └── 멀티유저 ref.txt
└── README.md
```

### 앱 비교

| 항목 | `multiref.py` | `multiusers.py` |
|------|---------------|-----------------|
| 사용자 인증 | 없음 | `user` 테이블 기반 로그인/회원가입 |
| 세션 저장 | Supabase | Supabase (`user_id` 분리) |
| 벡터 DB | Supabase pgvector | Supabase pgvector (`user_id` 분리) |
| LLM | `gpt-4o-mini` | `gpt-4o-mini` |
| 임베딩 | OpenAI `text-embedding-ada-002` | 동일 |
| 배포 | 로컬 / Streamlit Cloud | **Streamlit Cloud 배포 중** |

---

## 주요 기능

- **RAG**: PDF 업로드 → 청크 분할 → OpenAI 임베딩 → Supabase 벡터 저장 → 유사도 검색 답변
- **멀티세션**: 세션 저장 / 로드 / 삭제 / 화면 초기화 / vectordb 목록 조회
- **자동 저장**: 대화·PDF 처리 후 Supabase에 자동 반영
- **세션 제목 자동 생성**: 첫 Q&A를 LLM이 요약해 세션명 생성
- **스트리밍 답변**: 질문 입력 시 실시간으로 답변 표시
- **후속 질문 3개**: 답변 하단에 이어서 물어볼 질문 제안
- **멀티유저**: 사용자별 세션·메시지·벡터 데이터 완전 분리 (`user_id` FK)

---

## 사전 준비

### 환경 변수

저장소 루트 `fintech/.env` (로컬) 또는 Streamlit Cloud **Secrets**:

```env
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
```

Streamlit Cloud에서는 `st.secrets`가 우선 적용됩니다.

### Supabase DB 설정

**멀티세션** (`multiref.py`):

1. Supabase SQL Editor에서 `code/multi-session-ref.sql` 전체 실행

**멀티유저** (`multiusers.py`):

1. Supabase SQL Editor에서 `code/multiusers-ref.sql` 전체 실행
2. `user`, `chat_sessions`, `chat_messages`, `vector_documents` 테이블 생성 확인
3. 스키마 캐시 오류 시: `NOTIFY pgrst, 'reload schema';` 실행

> `multiusers-ref.sql`은 `multi-session-ref.sql`을 대체합니다. 멀티유저 앱 사용 시 **`multiusers-ref.sql`만** 실행하세요.

---

## 로컬 실행

```bash
cd 9.MultiService/code
pip install -r requirements.txt

# 멀티세션
streamlit run multiref.py

# 멀티유저
streamlit run multiusers.py
```

---

## Streamlit Cloud 배포

1. GitHub 저장소 연결
2. **Main file path**: `9.MultiService/code/multiusers.py`
3. **Secrets** 설정 (`OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`)
4. Supabase에서 `multiusers-ref.sql` 실행 완료 여부 확인
5. 배포 후 로그인 → PDF 업로드 → RAG 질의 → 세션 저장/로드 테스트

---

## 기술 스택

- **UI**: Streamlit
- **LLM / Embedding**: OpenAI (`gpt-4o-mini`, `text-embedding-ada-002`)
- **RAG**: LangChain (PyPDFLoader, RecursiveCharacterTextSplitter)
- **DB / Vector**: Supabase (PostgreSQL + pgvector)
- **인증**: 앱 자체 `user` 테이블 (PBKDF2-SHA256, Supabase Auth 미사용)

---

## 참고

- UI 스타일은 `4.ref/code/ref.py`를 참고했습니다.
- 벡터 저장 시 `file_name`을 명시적으로 INSERT하며, 검색은 RPC `match_vector_documents`로 `session_id`·`user_id` 필터링합니다.
