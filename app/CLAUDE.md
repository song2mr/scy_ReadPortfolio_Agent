# app/ — RAG 파이프라인 & Gradio UI

## 파이프라인 흐름

질문 → 라우터(RAG/NO_RAG/GENERAL) → Query Expansion → Hybrid Search(FAISS+BM25) → RRF 병합 → Cross-Encoder Rerank → LLM 생성 → 평가·재시도

## 주요 파일

- `rag.py` — RAG 파이프라인 전체. 공개 API: `get_answer()`, `get_answer_stream()`
- `rag_eval.py` — Faithfulness / Relevance LLM 평가 (1~5점)
- `app.py` — Gradio UI (비밀번호, 채팅, 프리셋 질문, 요약 다운로드, 키워드 통계)

## 후보자 정보

프롬프트에 주입되는 프로필(`PROFILE_BASIC`)과 쿼리 확장용 프로젝트 목록(`QUERY_EXPANSION_TOPICS`)은 `data/candidate_profile.py`에 정의. 후보자 변경 시 해당 파일만 수정.

## 핵심 config 파라미터

| 파라미터 | 설명 |
|---------|------|
| `OPENAI_MODEL` | 답변 생성 모델 |
| `OPENAI_MODEL_ROUTER` | 라우팅·쿼리 확장 (경량 모델) |
| `HYBRID_DENSE_K` / `HYBRID_SPARSE_K` | Dense/Sparse 검색 top-k |
| `RERANKER_ENABLED` / `RERANKER_TOP_N` | Cross-Encoder 설정 |
| `EVAL_RETRY_ENABLED` | 낮은 점수 시 자동 재시도 |
| `MAX_HISTORY_TURNS` | 대화 히스토리 최대 턴 |
| `RETRY_K_INCREMENT` | 재시도 시 k 증가량 |
