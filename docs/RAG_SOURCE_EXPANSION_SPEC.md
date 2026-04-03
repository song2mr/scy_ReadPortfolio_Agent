# RAG: 동일 source(파일) 청크 확장 스펙

## 목적

하이브리드 검색·Rerank는 **짧은 청크** 단위로 상위만 고릅니다. 포트폴리오를 **프로젝트당 PDF/DOCX 한 파일**로 두면, 검색에 걸린 파일의 **나머지 청크까지** 컨텍스트에 합쳐 답변이 끊기지 않게 합니다.

## 데이터 전제

- 한 프로젝트 ↔ 한 파일 권장 (`data/portfolio/`).
- 인덱스 빌드 후 `index/bm25_docs.pkl`에 전체 청크 리스트가 있어야 확장 가능(FAISS만 있는 구빌드는 확장 생략).
- 각 청크는 `metadata.chunk_kind`로 구분됩니다: **`summary`**(파일별 LLM 요약, 빌드 시 추가), **`body`**(본문 분할 청크). 동일 파일로 컨텍스트를 펼칠 때는 **요약 청크는 제외**하고 본문만 이어 붙입니다(중복·토큰 절감).

## 런타임: 요약으로 문서(source) 후보 → LLM `rag_scope`(BROAD / SINGLE)

인덱스에 `chunk_kind=summary`가 있고 `RAG_SUMMARY_ROUTING_ENABLED=true`이면:

1. **1단계**: Dense·BM25를 **요약 청크에만** 적용해, Rerank까지 거친 **요약 후보 목록**을 얻음.
2. **LLM `rag_scope`**: 질문이 **BROAD**(목록·전반·여러 프로젝트 개요) vs **SINGLE**(특정 한 프로젝트·주제 깊이)인지 분류.
3. **SINGLE**: 1단계에서 **가장 위 순위인 한 파일(`source`)**의 **본문 청크만** 컨텍스트에 넣음(고유 소스가 여러 개 Rerank에 보여도 무시).
4. **BROAD**: **요약 청크만** 본문은 넣지 않음. 기본(`RAG_BROAD_USE_ALL_SUMMARIES=true`)은 **인덱스에 있는 요약 전체**(소스당 1청크, 빌드 순서)를 넣고, `SOURCE_EXPANSION_MAX_CONTEXT_CHARS`로 길이 제한. 끄면 Rerank 상위만 `RAG_BROAD_MAX_*`로 제한.
5. 요약 인덱스가 없으면 예전처럼 **전체 청크 hybrid + source 확장**으로 폴백.

## (구) 동작 요약 — source 확장만 쓰는 경로

1. Hybrid + RRF → Rerank로 상위 청크 확정.
2. 그 청크들에 등장하는 `metadata["source"]`(없으면 `title`)를 **Rerank 순 첫 등장 순**으로 모음.
3. `SOURCE_EXPANSION_MAX_SOURCES`개까지만 확장 대상으로 삼음.
4. 각 source에 대해 `bm25_docs.pkl` 전역 순서에서 해당 source인 청크를 **문서 순서대로** 모음.
5. `SOURCE_EXPANSION_MAX_CHUNKS_PER_SOURCE`가 0이 아니면 파일당 청크 수를 자름.
6. `_format_docs`로 이어 붙이되, 합친 문자 수가 `SOURCE_EXPANSION_MAX_CONTEXT_CHARS`를 넘지 않도록 **앞에서부터** 청크를 포함하고 초과분은 버림.

## 설정 (`config.py`)

| 변수 | 의미 |
|------|------|
| `RAG_SUMMARY_ROUTING_ENABLED` | 요약 1차 검색·LLM `rag_scope` 기반 본문/요약 분기 on/off |
| `RAG_SUMMARY_DENSE_POOL` | FAISS에서 가져온 뒤 요약만 고를 때 초기 dense 후보 수 |
| `RAG_BROAD_USE_ALL_SUMMARIES` | `true`(기본)면 BROAD 시 인덱스 요약 전체(소스당 1개); `false`면 Rerank 기반 + 아래 max |
| `RAG_BROAD_MAX_SUMMARY_SOURCES` | 전체 요약 끔일 때 BROAD 최대 파일 수 |
| `RAG_BROAD_MAX_SUMMARY_CHUNKS` | 전체 요약 끔일 때 요약 청크 상한 |
| `SOURCE_EXPANSION_ENABLED` | (폴백·legacy 경로) 확장 on/off |
| `SOURCE_EXPANSION_MAX_SOURCES` | Rerank에 나온 서로 다른 파일 중 최대 몇 개까지 전체 청크 포함 |
| `SOURCE_EXPANSION_MAX_CHUNKS_PER_SOURCE` | 파일당 청크 상한(0 = 해당 제한 없음, 문자 상한만) |
| `SOURCE_EXPANSION_MAX_CONTEXT_CHARS` | 확장 포함 최종 컨텍스트 문자 상한 |

토큰 한도는 모델·API마다 다르므로, 문자 상한을 조정해 운영 환경에 맞출 것.

## "전체 포트폴리오 다 설명" 요청

이 스펙 **만으로**는 모든 파일을 한 번에 넣지 않습니다. 파일이 많으면 컨텍스트·비용 한계로 불가능에 가깝습니다.

권장 방향(별도 작업):

- **빌드 타임**: 파일별 **요약 청크**를 추가 인덱싱해, 포괄 질문은 요약 레이어를 먼저 검색.
- **제품 흐름**: 먼저 프로젝트 목록·한 줄 소개를 답하고, 사용자가 항목을 고르면 해당 파일만 확장해 상세 답변.

## 예외 / 한계

- 한 파일에 프로젝트가 여러 개 섞여 있으면 **파일 단위 확장은 과다 포함**이 될 수 있음 → 문서를 쪼개거나 추후 `project_id` 메타데이터 검토.
- `source` 문자열이 로더마다 달라 같이 파일인데 키가 다른 경우는 드물지만, 발생 시 빌드 단에서 `metadata` 정규화 검토.
