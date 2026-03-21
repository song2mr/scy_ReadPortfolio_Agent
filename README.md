---
title: 포트폴리오 RAG 에이전트
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# 포트폴리오 RAG 에이전트

PDF·Word 또는 Google Drive(문서·스프레드시트·PDF) 포트폴리오를 인덱싱해 두면, **인사·채용 담당자가 채팅으로 질문**할 수 있는 RAG(Retrieval-Augmented Generation) 에이전트입니다.  
**Gradio** UI, **LangChain**, **하이브리드 검색**(FAISS + BM25/Kiwi), **OpenAI** 답변 생성. 배포는 **Hugging Face Spaces**(Private 권장).

---

## 한눈에 보기 (비기술 요약)

| 구분 | 내용 |
|------|------|
| **무엇을 하나요?** | 지원자 문서를 미리 “검색 가능한 형태”로 만들어 두고, 질문이 오면 관련 문단을 찾아 그 근거로 답을 씁니다. |
| **누가 쓰나요?** | 채용·인사 담당자(또는 포트폴리오를 검토하는 사람). 지원자는 문서만 준비하면 됩니다. |
| **어떻게 쓰나요?** | 문서 넣기 → 인덱스 빌드 → 웹 채팅에서 질문. 답은 스트리밍으로 보이고, 필요하면 대화 요약을 파일로 받을 수 있습니다. |
| **핵심 흐름** | 질문 → (라우터가 RAG 필요 여부 판단) → 필요 시 **질문 재작성** → **벡터+키워드 검색** → **재순위** → LLM이 문서 근거로 답변 |

아키텍처 다이어그램(요청~응답, 하이브리드 검색): [`docs/RAG_ARCHITECTURE.md`](docs/RAG_ARCHITECTURE.md) · HTML 시각화: [`docs/architecture_diagram.html`](docs/architecture_diagram.html), [`docs/question_flow_diagram.html`](docs/question_flow_diagram.html)

---

## 기능 요약

- **3-way 라우터**: 질문을 한 번에 **RAG** / **NO_RAG**(인사·메타, 문서 없이 답 가능) / **GENERAL**(포트폴리오 무관)으로 분류. 대화 히스토리를 참고해 맥락 있는 분류.
- **Query Expansion**: 질문만 LLM으로 재작성(단일 → 1줄, 포괄적 → 여러 하위 질문). 쿼리별 검색 후 **RRF**로 병합.
- **RAG 답변**: **Hybrid Search**(FAISS Dense + BM25/Kiwi Sparse, RRF) → **Reranker**(Cross-Encoder) → OpenAI 답변. 참고 문단 접기 표시.
- **스트리밍**: 답변이 토큰 단위로 실시간 출력.
- **대화 요약 다운로드**: MD / PDF(한글 실패 시 TXT)로 대화 요약·전체 내역 저장.
- **질문 키워드 통계**: 경력, 강점, 프로젝트 등 키워드 언급 횟수 표시.
- **평가 후 재시도**: Faithfulness/Relevance 점수가 낮으면 k를 늘려 1회 재검색·재생성(`config.py`로 on/off).

---

## 빠른 시작

### 1. 환경

- **Python 3.13+** (`pyproject.toml` 기준). 로컬 개발은 **[uv](https://github.com/astral-sh/uv)** 권장.
- `.env`에 `OPENAI_API_KEY=sk-...` 설정

```bash
uv sync

cp .env.example .env
# .env에 OPENAI_API_KEY 입력
```

### 2. 포트폴리오 넣기

- **로컬**: `data/portfolio/`에 PDF 또는 Word(`.docx`)를 둡니다.
- **Google Drive**: `.env`에 `GOOGLE_DRIVE_FOLDER_ID`를 설정하면 해당 폴더의 Google Docs·스프레드시트·PDF를 사용합니다. ([구글 드라이브 사용](#구글-드라이브-사용))

### 3. 인덱스 빌드

```bash
uv run python scripts/build_index.py
```

- `index/`에 FAISS(`index.faiss`, `index.pkl`)와 BM25(Kiwi) 인덱스(`bm25_corpus.pkl`, `bm25_docs.pkl`)가 생성됩니다.
- 문서를 바꾼 뒤에는 이 명령을 다시 실행하세요.
- `GOOGLE_DRIVE_FOLDER_ID`가 있으면 드라이브에서, 없으면 `data/portfolio/`에서 로드합니다.

### 4. 앱 실행

프로젝트 **루트**에서:

```bash
uv run python app.py
```

또는

```bash
uv run python -m app.app
```

브라우저에서 채팅·추천 질문·대화 초기화·요약 다운로드(MD/PDF)를 사용합니다.

> `main.py`는 패키지 스캐폴드용 플레이스홀더이며, 앱 진입점은 위 `app.py`입니다.

---

## 설정 (`config.py`)

| 항목 | 설명 | 기본값(참고) |
|------|------|----------------|
| `OPENAI_MODEL` | 답변 생성(RAG·NO_RAG·GENERAL) | `gpt-5` — 실제 계정에서 쓰는 모델명으로 변경 |
| `OPENAI_MODEL_ROUTER` | 라우터 분류용 | `gpt-4o-mini` |
| `HYBRID_DENSE_K` / `HYBRID_SPARSE_K` | 쿼리당 Dense·Sparse top-k | `7` / `7` |
| `HYBRID_MERGE_TOP_N` | RRF 병합 후 Reranker 전 상위 개수 | `15` |
| `RERANKER_ENABLED` / `RERANKER_TOP_N` | Cross-Encoder 재순위화 | `True` / `8` |
| `QUERY_EXPANSION_MAX_SUB_QUERIES` | 포괄적 질문 시 하위 질문 상한 | `5` |
| `EVAL_RETRY_ENABLED` | 평가 후 재시도 | `True` |
| `EVAL_MIN_FAITHFULNESS` / `EVAL_MIN_RELEVANCE` | 미만이면 k 증가 후 1회 재시도 | `3` / `3` |
| `EMBEDDING_MODEL` | FAISS 임베딩 | `jhgan/ko-sroberta-multitask` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 인덱스 빌드 시 청킹 | `800` / `150` |

라우터는 분류만 하므로 `OPENAI_MODEL_ROUTER`를 가벼운 모델로 두면 비용·지연을 줄일 수 있습니다.

---

## 환경 변수 (`.env` / Space)

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | 필수(로컬). Space는 **Secrets**에 등록 |
| `GOOGLE_DRIVE_*` | 선택 — 드라이브에서 빌드 시 ([아래](#구글-드라이브-사용)) |
| `LANGSMITH_*` | 선택 — 트레이싱 ([아래](#langsmith-트레이싱)) |
| `DEBUG_RAG=1` | 선택 — 터미널에 `[RAG]` 디버그 로그 |
| `HF_INDEX_DATASET` | 선택 — Space 등에서 로컬 `index/`가 없을 때 HF Dataset/Model ID로 인덱스 다운로드 |
| `HF_TOKEN` | 선택 — Private Dataset/Model 접근 시 |

자세한 예시는 `.env.example`을 참고하세요.

---

## 구글 드라이브 사용

데이터를 **구글 드라이브 폴더**에 두고 청킹·임베딩에 사용할 수 있습니다.

1. [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성 후 **Google Drive API** 사용 설정.
2. **OAuth 2.0** 데스크톱 앱 클라이언트로 `credentials.json` 발급.
3. `credentials.json`을 `~/.credentials/credentials.json`에 두거나 `.env`의 `GOOGLE_DRIVE_CREDENTIALS_PATH`로 지정. 첫 실행 시 브라우저 동의 후 `token.json` 생성.
4. 폴더 URL `.../folders/xxxx`의 `xxxx`를 `GOOGLE_DRIVE_FOLDER_ID`에 설정.
5. `uv run python scripts/build_index.py` 실행.

지원 형식: Google Docs, Google 스프레드시트, PDF.

---

## LangSmith (트레이싱)

`.env`에 다음을 설정하면 LangChain 실행이 [LangSmith](https://smith.langchain.com/)에 기록됩니다.

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=lsv2_pt_...`
- (선택) `LANGSMITH_PROJECT=scy-rag`

`app.py`는 import 순서상 LangSmith를 가능하면 먼저 설정합니다.

---

## 디버깅

`.env`에 `DEBUG_RAG=1` 후 터미널에서 앱을 실행하면 라우터·검색·스트리밍 단계별 `[RAG]` 로그가 출력됩니다.

---

## 지원자 맞춤화

- 첫 인사·추천 질문·키워드 통계: `app/app.py`의 `FIRST_MESSAGE`, `PRESET_QUESTIONS`, `KEYWORDS`
- RAG/NO_RAG 시스템 프롬프트·지원자 요약: `app/rag.py`의 `PROFILE_BASIC`, `RAG_SYSTEM`, `NO_RAG_PORTFOLIO_SYSTEM`, `GENERAL_SYSTEM`, `QUERY_EXPANSION_PORTFOLIO_TOPICS`

---

## 대화 히스토리(맥락) 반영

별도 DB 없이 **매 요청마다 최근 N턴**을 프롬프트에 넣습니다.

1. Gradio에서 `history_pairs`가 `get_answer` / `get_answer_stream`으로 전달됩니다.
2. 라우터는 최근 **5턴**을 짧게 잘라 `<recent_conversation>`에 넣습니다.
3. 답변 생성은 최근 **5턴**을 `HumanMessage` / `AIMessage`로 `MessagesPlaceholder("chat_history")`에 넣습니다.

N은 `app/rag.py`의 `_pairs_to_messages(max_turns=5)`, `_format_history_for_router(max_turns=5)`에서 조정할 수 있습니다.

---

## 프로젝트 구조

```
├── app.py                 # Gradio 진입점 (Hugging Face Spaces / 로컬)
├── main.py                # 패키지 플레이스홀더 (앱 진입점 아님)
├── config.py              # 경로, 청킹, 검색, LLM, Reranker 등
├── pyproject.toml         # uv 의존성 (Python >=3.13)
├── requirements.txt       # HF Spaces용 pip 목록
├── app/
│   ├── app.py             # UI (build_ui), 이벤트
│   ├── rag.py             # 라우터, Query Expansion, Hybrid Search, get_answer / stream
│   └── rag_eval.py        # Faithfulness / Relevance 평가
├── scripts/
│   ├── build_index.py     # 로컬 또는 드라이브 → 청킹 → index/
│   └── evaluate_rag.py    # RAG 평가 스크립트 (선택)
├── data/portfolio/        # PDF·DOCX (로컬)
├── index/                 # FAISS + BM25 (빌드 후 생성)
├── docs/
│   ├── RAG_ARCHITECTURE.md
│   ├── 코드_동작_설명.md
│   └── *.html             # 아키텍처·질문 흐름 다이어그램
├── notebooks/             # 실험용 (예: test_rag.ipynb)
└── .env                   # API 키 등 (git 제외)
```

---

## RAG 평가 (선택)

```bash
uv run python scripts/evaluate_rag.py
```

인덱스가 있어야 합니다. `app/rag_eval.py`의 `evaluate_response_from_docs`를 코드에서 직접 호출할 수도 있습니다.

---

## 배포 (Hugging Face Spaces)

무료 계정으로 **Private** Space를 만들어 다른 사람이 보지 못하게 배포할 수 있습니다. (무료 private 저장 100GB 이내, CPU Basic 하드웨어 무료.)

### 1. Space 만들기

1. [huggingface.co](https://huggingface.co) 로그인(또는 가입).
2. [Spaces](https://huggingface.co/spaces) → **Create new Space**.
3. **SDK**: **Gradio**, **Visibility**: **Private** 권장.

### 2. Space에 넣을 파일

로컬에서 아래에 맞춰 필요한 파일만 Space repo에 푸시합니다. **`.env`는 넣지 않고**, API 키는 Space **Secrets**로 넣습니다.

| 포함 여부 | 경로 | 설명 |
|----------|------|------|
| ✅ | `app.py` | 루트 진입점 (Gradio `demo`). |
| ✅ | `config.py` | 경로, 청킹, 모델 등 |
| ✅ | `app/` | `app.py`, `rag.py`, `rag_eval.py` |
| ✅ | `scripts/` | `build_index.py` (선택) |
| ✅ | `index/` | FAISS + BM25(`bm25_corpus.pkl`, `bm25_docs.pkl`). **로컬에서 빌드 후** 커밋 또는 HF Dataset 사용 |
| ✅ | `requirements.txt` | Space용 `pip` 의존성 |
| ✅ | (선택) `data/portfolio/` | 인덱스만 repo에 있으면 생략 가능 |
| ❌ | `.env` | 푸시 금지 — **Settings → Secrets** |
| ❌ | `token.json`, `credentials.json` | 구글 드라이브용 — 로컬 전용 |

- **진입점**: Space는 기본적으로 루트 `app.py`를 실행합니다.

### 3. Secrets 설정

1. Space 저장소 → **Settings** → **Variables and secrets**.
2. **New secret**: 이름 `OPENAI_API_KEY`, 값 `sk-...`.
3. (선택) 드라이브를 Space에서 쓸 경우 `GOOGLE_DRIVE_FOLDER_ID` 등 — 보통은 **인덱스만** 올리는 방식을 권장.

### 4. 푸시 후 확인

- `git push` 후 자동 빌드·실행.
- 무료 CPU Basic은 미사용 시 **슬립** — 첫 접속 시 기동이 느릴 수 있음.
- Private이면 URL을 아는 사람만 접근(본인 계정 기준).

### 5. Space에 인덱스 넣기 (HF Dataset)

Git이 `index.faiss` 등 바이너리 푸시를 거절할 수 있으면, **Dataset**에 올리고 시작 시 다운로드합니다.

1. 로컬: `uv run python scripts/build_index.py` → `index/` 생성.
2. [huggingface.co/datasets](https://huggingface.co/datasets) → **Create new dataset** (Private 가능).
3. **Files** → 업로드: `index.faiss`, `index.pkl`, `bm25_corpus.pkl`, `bm25_docs.pkl` (Dataset **루트** 또는 하위 폴더 — 코드가 `index.faiss` 위치를 탐색).
4. Space **Variables**: `HF_INDEX_DATASET` = `본인아이디/데이터셋이름`. Private Dataset은 같은 HF 계정 Space에서 접근.
5. (선택) Private 접근용 **`HF_TOKEN`**을 Secrets에 추가.

### 6. 배포 요약표

| 항목 | 내용 |
|------|------|
| Private | Space 생성 시 **Visibility: Private** 선택 가능(무료). |
| 비용 | 저장 100GB·CPU Basic 무료(추가 하드웨어는 유료). |
| 진입점 | 루트 `app.py`. |
| API 키 | **Settings → Secrets**에 `OPENAI_API_KEY`. |
| 인덱스 | Dataset에 파일 업로드 후 `HF_INDEX_DATASET`, 필요 시 `HF_TOKEN`. |

---

## 관련 문서

- 코드 흐름·함수 역할: [`docs/코드_동작_설명.md`](docs/코드_동작_설명.md)
- RAG 파이프라인 구조(Mermaid): [`docs/RAG_ARCHITECTURE.md`](docs/RAG_ARCHITECTURE.md)
