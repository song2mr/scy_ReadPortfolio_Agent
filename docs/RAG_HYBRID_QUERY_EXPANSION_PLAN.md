# RAG 고도화 계획: Hybrid 구조, Query Expansion, Hybrid Search

## 1. 현재 구조 요약

| 구분 | 내용 |
|------|------|
| **라우팅** | LLM 기반 3-way: RAG / NO_RAG / GENERAL |
| **검색** | 단일 경로: **Dense only** (FAISS + `jhgan/ko-sroberta-multitask`) |
| **재순위화** | Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3), 상위 N개만 LLM에 전달 |
| **인덱스 빌드** | `scripts/build_index.py`: PDF/DOCX → 청킹 → 임베딩 → FAISS 저장 |

**데이터 흐름 (RAG 경로):**  
`질문` → 라우터 → **retriever(질문)** → Reranker → context → LLM → 답변

---

## 2. 적용할 세 가지 개선

### 2.1 Hybrid 구조 (시스템 관점)

여기서 "Hybrid"는 **검색 방식**을 의미합니다.

- **현재**: Dense(벡터) 검색만 사용.
- **목표**: Dense + Sparse(키워드/BM25)를 함께 쓰는 **Hybrid Search**로 확장.  
  (라우팅은 그대로 두고, "RAG일 때 검색 단계만" Hybrid로 바꾸는 방식.)

즉, **2.3 Hybrid Search**를 적용하면 "Hybrid 구조"가 됩니다.  
별도로 "라우팅 구조를 Hybrid로 바꾼다"는 요구가 없다면, **구조 변경 = Hybrid Search 도입**으로 보면 됩니다.

---

### 2.2 Query Expansion (질의 확장)

**목적**: 사용자 질문이 짧거나 애매할 때, 동의어·유의어·재서술을 추가해 검색 품질을 높임.

**진입 위치 (현재 구현)**:  
`질문` → **(LLM: 질문만으로 재작성, Few-shot)** → 확장된 질문(들) → Retriever → …  
※ 1차 소량 검색·문서 발췌는 사용하지 않음 (질문만으로 Multi-Query 스타일 재작성).

- **Few-shot**: 프롬프트에 "원본 질문 → 재작성된 질문" 예시를 넣어, 단일/포괄적 질문에 따라 1줄 또는 2~4개 하위 질문으로 재작성하도록 유도.

**선택지 요약**

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A. LLM 기반** | LLM에게 "이 질문을 2~3개로 확장/재서술해줘" 요청 | 맥락 반영, 자연어에 강함 | API 비용·지연, 확장 품질 변동 |
| **B. 동의어/키워드** | 미리 정의한 용어 사전으로 키워드 추가 | 빠름, 비용 없음, 예측 가능 | 도메인별 사전 관리 필요 |
| **C. 하이브리드** | B로 키워드 추가 후, 필요 시에만 LLM 확장 | 비용·지연 절충 | 구현 복잡도 증가 |

**구현 시 고려**

- 확장된 질문이 **1개**가 아니라 **여러 개**일 때: 각각 검색 후 결과 **병합**(RRF 등)할지, **하나로 합쳐서** 한 번만 검색할지 결정 필요.
- RAG 경로에서만 적용할지, 라우터 입력에는 원본 질문만 쓸지(권장: 라우터는 원본, 검색만 확장).

**참고: LLM 확장 vs 동의어 사전**

- **동의어/키워드 사전**: 코드에 정의된 고정 매핑(예: `{"강점": ["역량", "강점"]}`)으로 단어 치환. LLM 호출 없음.
- **LLM 확장**: 프롬프트로 모델에게 "질문을 재작성해줘"라고 요청하는 방식. 아래 `reformulation_template` 같은 프롬프트는 **LLM 확장**에 해당함.

---

### 2.3 Hybrid Search (RAG 내 검색)

**목적**: Dense(의미) + Sparse(키워드) 검색을 결합해, 키워드가 정확히 맞는 문단도 놓치지 않고 의미 유사 문단도 가져오기.

**현재**: FAISS(dense)만 사용.  
**목표**: Dense 결과 + BM25(또는 다른 sparse) 결과를 합쳐서 하나의 문서 목록으로 만든 뒤, 기존처럼 Reranker로 재순위화.

**구성 요소**

1. **Sparse 인덱스**
   - BM25 등 키워드 검색용 인덱스.
   - 청크 텍스트(`page_content`)로 빌드.  
   - 저장: `index/` 디렉터리에 `.pkl` 또는 별도 파일(e.g. `bm25_index.pkl`).

2. **검색 단계**
   - Dense: 기존 FAISS retriever로 top-k₁ 가져오기.
   - Sparse: BM25로 동일(또는 다른) top-k₂ 가져오기.
   - **결과 병합**: Reciprocal Rank Fusion (RRF) 또는 점수 가중 평균.  
     → 중복 제거 후 상위 N개를 Reranker에 입력.

3. **인덱스 빌드**
   - `scripts/build_index.py`에서 FAISS 생성 시, **동일 청크 리스트**로 BM25 인덱스도 생성해 함께 저장.

**선택지 요약**

| 항목 | 옵션 | 비고 |
|------|------|------|
| Sparse 엔진 | BM25 (rank_bm25 등) | 한국어는 형태소 분석기(Okt/Kiwi 등) 사용 권장 |
| 병합 방식 | RRF / 가중 점수 | RRF가 구현 단순·튜닝 적음 |
| k 값 | Dense k, Sparse k 각각 설정 | 예: 각 10개 → 병합 후 15~20개 → Reranker → 4개 |

---

## 3. 구현 단계 제안 (한 번에 하지 말고)

| 단계 | 내용 | 산출물 |
|------|------|--------|
| **1** | Hybrid Search 도입 (BM25 인덱스 + 병합 로직) | `build_index.py`에 BM25 저장, `rag.py`에 dense+sparse 병합 |
| **2** | Query Expansion 도입 (선택한 방식으로) | `rag.py`에 확장 단계 추가, config 옵션 |
| **3** | (선택) 튜닝·평가 | k, RRF 파라미터, 확장 on/off에 따른 품질 비교 |

먼저 **1단계만** 적용해도 "Hybrid 구조 + Hybrid Search"가 되고,  
이후 **2단계**에서 Query Expansion을 끼워 넣으면 됩니다.

---

## 4. 사용자가 정해야 할 것 (결정 사항)

아래는 **당신이 선택/결정**해야 할 항목입니다. 결정 후 계획서를 기준으로 단계별 구현을 진행하면 됩니다.

### 4.1 Query Expansion

1. **확장 방식**
   - [x] **A. LLM 기반** (비용·지연 감수, 품질 우선) ← **확정**
   - [ ] **B. 동의어/키워드 사전** (무비용, 도메인 사전 관리)
   - [ ] **C. 하이브리드** (사전 + 필요 시 LLM)

2. **확장 결과 사용**
   - [x] 확장된 질문 **여러 개**로 각각 검색 후 **결과 병합**(RRF 등) ← **확정**
   - [ ] 확장된 질문을 **하나의 문장으로 합쳐서** 한 번만 검색

3. **적용 범위**
   - [x] RAG로 분류된 질문에만 적용 (권장)
   - [x] 라우터에는 원본 질문만 사용 (권장)

### 4.2 Hybrid Search

4. **Sparse 쪽 한국어 처리**
   - [x] **형태소 분석기 사용** — **Kiwi** 사용 ← **확정**
   - [ ] **공백/단어 단위만** (형태소 없이) → 구현 단순, 한국어 품질 다소 떨어질 수 있음

5. **병합 방식**
   - [x] **RRF** (Reciprocal Rank Fusion) ← **확정**
   - [ ] **가중 점수** (Dense 점수·Sparse 점수 정규화 후 가중합) — 가중치 튜닝 필요

6. **각 경로 k 값** ← **확정 (제안대로)**
   - [x] Dense top-k: **10**
   - [x] Sparse top-k: **10**
   - [x] 병합 후 Reranker 전 상위 개수: **15** (기존 `RETRIEVE_K_INITIAL` 유지)

### 4.3 운영/설정

7. **기능 플래그**
   - [x] **상시 on** — Query Expansion, Hybrid Search 둘 다 항상 사용 (off 옵션 없음) ← **확정**
   - ~~Query Expansion on/off~~ / ~~Hybrid Search on/off~~ (폴백 불필요)

8. **인덱스 호환** ← **확정**
   - 기존 FAISS만 있는 환경: 기존처럼 동작 (필수)
   - [x] BM25 인덱스가 없으면 **자동으로 Dense만 사용** (권장대로 적용)

---

## 5. 계획서 정리

- **Hybrid 구조**: RAG 검색을 **Dense + Sparse Hybrid Search**로 바꾸는 것으로 정의.
- **Query Expansion**: 질문을 확장한 뒤 검색에만 사용 (라우터는 원본 유지).
- **Hybrid Search**: FAISS + BM25 병합 후 기존 Reranker 유지.

위 **4.1~4.3**에서 체크한 선택지를 알려주시면, 그에 맞춰 **1단계(Hybrid Search)**부터 구체적인 코드 변경 계획(어디를 수정할지, config 추가안)을 이어서 작성하겠습니다.

---

## 6. 확정된 선택 (2025-03 기준)

| 항목 | 선택 |
|------|------|
| **Query Expansion 방식** | **LLM 기반** (프롬프트로 재작성 요청) |
| **Query Expansion 사용법** | 확장된 질문 **각각 검색 후 결과 병합**(RRF 등) |
| **Query Expansion** | **질문만** LLM 재작성 (1차 검색 없음), Few-shot. 단일 → 1줄, 포괄적 → 2~4개 하위 질문 |
| **Query Expansion 프롬프트** | `REFORMULATION_TEMPLATE` (질문만 입력, Few-shot 포함) |
| **Hybrid Search 한국어** | **Kiwi** 형태소 분석기 사용 |
| **Hybrid Search 병합** | **RRF** (Reciprocal Rank Fusion) |
| **Hybrid Search k 값** | Dense top-k **10**, Sparse top-k **10**, 병합 후 Reranker 전 **15** |
| **인덱스 호환** | BM25 없으면 자동으로 Dense만 사용 |
| **기능 플래그** | **상시 on** (Query Expansion, Hybrid Search 둘 다 항상 사용, off 옵션 없음) |

### Query Expansion 흐름 (확정)

1. **LLM 재작성**: 원본 질문만 넣고 Few-shot으로 재작성 (단일 → 1줄, 포괄적 → 2~4개 하위 질문).
2. **검색**: [원본, 재작성1, 재작성2, ...] 각 쿼리마다 Hybrid Search → 쿼리당 **HYBRID_MERGE_TOP_N개**(예: 15개) 문서 → 여러 랭킹을 RRF로 병합.

- 라우터에는 **원본 질문**만 사용. Query Expansion은 검색 단계에서만 (1차 검색 없이 질문만 재작성).

---

### Query Expansion용 LLM 프롬프트 (확정) — 문서 맥락 + Few-shot

```python
# 1차 검색으로 얻은 document_snippets 를 {context} 로 넣음.
# Few-shot 예시로 포트폴리오 Q&A 스타일의 재작성 패턴을 제시.

reformulation_template = """당신은 지원자 포트폴리오 문서를 검색하는 시스템을 위한 쿼리 재작성 전문가입니다.
아래 [참고 문서]는 사용자 질문과 관련될 수 있는 문서 발췌입니다. 이 문서에 나오는 표현·키워드·개념을 반영하여, 검색 성능이 올라가도록 질문을 재작성해 주세요.

재작성 시 다음을 적용하세요:
1. 문서에 등장하는 동의어·유사 표현 추가
2. 더 구체적인 키워드 포함 (문서 용어 우선)
3. 관련 개념 확장 (문서 맥락 내에서)

---
## Few-shot 예시

### 예시 1
[참고 문서]
- 경력: 에코마케팅 인턴, 이엠넷 R&D본부. 강점: 로그 트래킹, ETL 파이프라인, ML·RAG 활용.

[원본 질문]
이 사람 뭐 해?

[재작성된 질문]
이 지원자의 주요 경력, 현재 직무와 담당 업무, 그리고 강점(로그 트래킹, ETL, RAG 등)을 알려주세요.

### 예시 2
[참고 문서]
- 활동: QMS 데이터 분석 학회, KCI 논문 1저자, 데이터청년캠퍼스. 소프트웨어: Python, R, SQL, Looker Studio.

[원본 질문]
프로젝트 알려줘

[재작성된 질문]
데이터 분석 관련 프로젝트, 학회 활동, 논문 등 연구 경험과 사용 기술(Python, R, SQL 등)을 구체적으로 알려주세요.

---
## 실제 작업

[참고 문서]
{context}

[원본 질문]
{question}

[재작성된 질문]
"""
```

- **입력**: `context` = 1차 소량 검색으로 얻은 문서 발췌(문단 요약 또는 `page_content` 일부), `question` = 사용자 원본 질문.
- **출력**: 재작성된 질문 1개 (필요 시 파싱 규칙으로 여러 개로 분리 가능).
- 확장 결과가 여러 개이면: 원본 + 재작성 질문 각각 검색 후 RRF로 병합.

---

### HF(Hugging Face) 업로드 시 포함할 인덱스 파일

`index/` 폴더를 Dataset 또는 Model로 업로드할 때 아래 파일이 **모두** 있으면 Hybrid Search + Query Expansion이 동작합니다.

| 파일 | 용도 |
|------|------|
| `index.faiss` | Dense 검색 (FAISS) |
| `index.pkl` | FAISS 메타데이터 |
| `bm25_corpus.pkl` | BM25용 토큰화된 코퍼스 (Kiwi) |
| `bm25_docs.pkl` | BM25용 Document 리스트 |

- `bm25_*.pkl` 이 없으면 자동으로 **Dense만** 사용합니다 (기존 FAISS만 있어도 동작).
- 인덱스 빌드: `uv run python scripts/build_index.py` → 위 네 파일이 모두 `index/` 에 생성됩니다.
