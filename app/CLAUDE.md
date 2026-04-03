# app/ — RAG 파이프라인 & Gradio UI

## 파이프라인 흐름 (RAG 경로)

질문 → **라우터** (RAG / NO_RAG / GENERAL) → **Query Expansion** (하위 쿼리; RRF 병합용) → **`rag_scope` LLM** (BROAD | SINGLE) → **Retrieve** (요약 우선 hybrid + Rerank; scope에 따라 요약 전체 vs 한 파일 본문) → **LLM 생성** → (선택) 평가·재시도

- **BROAD**: 컨텍스트는 **요약만**. 기본은 인덱스 **요약 전체**(소스당 1청크) + `SOURCE_EXPANSION_MAX_CONTEXT_CHARS`.
- **SINGLE**: Rerank **1순위 `source`** 의 **본문 청크 전부**(요약 제외).
- 회상·나열 패턴은 `RAG_SCOPE_BROAD_HEURISTIC` 으로 BROAD 고정 가능.
- 검색·쿼리 확장 상세·다이어그램: 루트 `docs/chain_flow.html`

## 주요 파일

- `rag.py` — RAG 파이프라인 전체. 공개 API: `get_answer()`, `get_answer_stream()`
- `portfolio_origins.py` — `portfolio_origins.yaml` 로드·경로별 `portfolio_origin` 해석 (빌드·스크립트에서 사용)
- `rag_eval.py` — Faithfulness / Relevance LLM 평가 (1~5점)
- `app.py` — Gradio UI (비밀번호, 채팅, 프리셋 질문, 요약 다운로드, 키워드 통계)

## 후보자 정보

프롬프트에 주입되는 프로필(`PROFILE_BASIC`)과 쿼리 확장용 프로젝트 목록(`QUERY_EXPANSION_TOPICS`)은 `data/candidate_profile.py`에 정의. 후보자 변경 시 해당 파일만 수정.

## 핵심 config 파라미터

| 파라미터 | 설명 |
|---------|------|
| `OPENAI_MODEL` | 답변 생성 모델 |
| `OPENAI_MODEL_ROUTER` | 라우팅·쿼리 확장·`rag_scope` (경량 모델 권장) |
| `RAG_SUMMARY_ROUTING_ENABLED` | 요약 1차 검색 + BROAD/SINGLE 분기 사용 |
| `RAG_BROAD_USE_ALL_SUMMARIES` | BROAD 시 인덱스 요약 전체 vs Rerank 상위만 |
| `RAG_SCOPE_BROAD_HEURISTIC` | 나열·회상형 질문 BROAD 고정 |
| `HYBRID_DENSE_K` / `HYBRID_SPARSE_K` | Dense/Sparse 검색 top-k |
| `RERANKER_ENABLED` / `RERANKER_TOP_N` | Cross-Encoder 설정 |
| `SOURCE_EXPANSION_MAX_CONTEXT_CHARS` | 최종 컨텍스트 문자 상한 |
| `EVAL_RETRY_ENABLED` | 낮은 점수 시 자동 재시도 |
| `MAX_HISTORY_TURNS` | 대화 히스토리 최대 턴 |
| `RETRY_K_INCREMENT` | 재시도 시 k 증가량 |
