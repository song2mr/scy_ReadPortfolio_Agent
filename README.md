---
title: 포트폴리오 RAG 에이전트
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">

# 📄 포트폴리오 RAG 에이전트

**인사 담당자를 위한 AI 포트폴리오 Q&A 챗봇**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.5+-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-412991?logo=openai&logoColor=white)](https://openai.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

PDF/Word 포트폴리오를 인덱싱해 두면, 채용 담당자가 **채팅으로 질문**할 수 있는 RAG 에이전트입니다.

[빠른 시작](#-빠른-시작) · [아키텍처](#-아키텍처) · [설정](#-설정-configpy) · [배포](#-배포-hugging-face-spaces)

</div>

---

## 🔍 한눈에 보기

| 구분 | 내용 |
|:----:|------|
| **무엇을?** | 지원자 문서를 검색 가능한 형태로 만들어 두고, 질문에 관련 문단을 찾아 근거 기반 답변 |
| **누가?** | 채용·인사 담당자 (지원자는 문서만 준비) |
| **어떻게?** | 문서 등록 → 인덱스 빌드 → 웹 채팅 질문 → 스트리밍 답변 + 대화 요약 다운로드 |

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **3-way 라우터** | RAG / NO_RAG / GENERAL 자동 분류 (대화 맥락 참고) |
| **Query Expansion** | 질문을 LLM으로 재작성, 포괄적 질문은 하위 쿼리로 분할 |
| **Hybrid Search** | FAISS(Dense) + BM25/Kiwi(Sparse) → RRF 병합 |
| **Cross-Encoder Reranker** | `bge-reranker-v2-m3`로 정밀 재순위화 |
| **스트리밍 답변** | 토큰 단위 실시간 출력 |
| **평가·재시도** | Faithfulness/Relevance 낮으면 k 늘려 1회 재검색 |
| **대화 요약 다운로드** | MD / PDF(한글 지원) / TXT |
| **키워드 통계** | 경력·강점·프로젝트 등 언급 빈도 시각화 |

---

## 🏗 아키텍처

```
질문 입력
    │
    ▼
┌─────────────────────┐
│   라우터 (3-way)     │  ← gpt-4o-mini
│  RAG / NO_RAG / GEN │
└────┬───────┬────┬───┘
     │       │    │
   [RAG]  [NO_RAG] [GENERAL]
     │       │    │
     ▼       │    │
┌──────────┐ │    │
│  Query   │ │    │
│ Expansion│ │    │
└──────────┘ │    │
     │       │    │
     ▼       │    │
┌──────────────────┐  │
│  Hybrid Search   │  │
│ FAISS + BM25/Kiwi│  │
│    ↓ RRF 병합     │  │
└──────────────────┘  │
     │       │    │
     ▼       │    │
┌──────────┐ │    │
│ Reranker │ │    │
│(Cross-Enc)│ │    │
└──────────┘ │    │
     │       │    │
     ▼       ▼    ▼
┌──────────────────────┐
│    LLM 답변 생성      │  ← gpt-5
│  (스트리밍 + 평가)     │
└──────────────────────┘
```

> 상세 다이어그램: [`docs/RAG_ARCHITECTURE.md`](docs/RAG_ARCHITECTURE.md) · [`docs/architecture_diagram.html`](docs/architecture_diagram.html)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# uv 사용 권장 (Python 3.13+)
uv sync
cp .env.example .env
# .env에 OPENAI_API_KEY=sk-... 입력
```

### 2. 포트폴리오 등록

`data/portfolio/`에 PDF 또는 Word(`.docx`)를 넣으세요.

> Google Drive도 지원합니다 → [구글 드라이브 사용](#-구글-드라이브-사용)

### 3. 인덱스 빌드

```bash
uv run python scripts/build_index.py
```

`index/`에 FAISS + BM25(Kiwi) 인덱스가 생성됩니다.

### 4. 앱 실행

```bash
uv run python app.py
```

브라우저에서 채팅, 추천 질문, 대화 요약 다운로드를 사용할 수 있습니다.

---

## ⚙ 설정 (`config.py`)

### LLM

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_MODEL` | `gpt-5` | 답변 생성 모델 |
| `OPENAI_MODEL_ROUTER` | `gpt-4o-mini` | 라우터·쿼리 확장 (경량) |

### 검색

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `HYBRID_DENSE_K` / `HYBRID_SPARSE_K` | `7` / `7` | 쿼리당 Dense·Sparse top-k |
| `HYBRID_MERGE_TOP_N` | `15` | RRF 병합 후 상위 개수 |
| `RERANKER_ENABLED` / `RERANKER_TOP_N` | `True` / `8` | Cross-Encoder 재순위화 |
| `QUERY_EXPANSION_MAX_SUB_QUERIES` | `5` | 하위 질문 최대 개수 |
| `EMBEDDING_MODEL` | `jhgan/ko-sroberta-multitask` | 한국어 임베딩 |

### 평가·재시도

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `EVAL_RETRY_ENABLED` | `True` | 평가 후 자동 재시도 |
| `EVAL_MIN_FAITHFULNESS` / `EVAL_MIN_RELEVANCE` | `3` / `3` | 이 점수 미만 시 재시도 |
| `RETRY_K_INCREMENT` | `5` | 재시도 시 k 증가량 |

### 기타

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | 인덱스 빌드 시 청킹 |
| `MAX_HISTORY_TURNS` | `5` | LLM에 전달하는 대화 히스토리 최대 턴 |

---

## 🔑 환경 변수

| 변수 | 필수 | 설명 |
|------|:----:|------|
| `OPENAI_API_KEY` | **필수** | OpenAI API 키 |
| `APP_PASSWORD` | 선택 | 미설정 시 `dev`로 입장 |
| `HF_INDEX_DATASET` | 선택 | HF Spaces에서 인덱스 다운로드 |
| `HF_TOKEN` | 선택 | Private Dataset 접근 |
| `DEBUG_RAG=1` | 선택 | 터미널 디버그 로그 |
| `LANGSMITH_API_KEY` | 선택 | LangSmith 트레이싱 |
| `LANGSMITH_TRACING=true` | 선택 | 트레이싱 활성화 |
| `GOOGLE_DRIVE_FOLDER_ID` | 선택 | 구글 드라이브 폴더 ID |

> 전체 예시: `.env.example` 참조

---

## 🧑‍💼 후보자 변경 방법

후보자 정보는 코드가 아닌 **데이터 파일**에서 관리합니다:

| 파일 | 내용 |
|------|------|
| `data/candidate_profile.py` | `PROFILE_BASIC` (경력·학력·강점), `QUERY_EXPANSION_TOPICS` (프로젝트 목록) |
| `data/portfolio/` | PDF·DOCX 포트폴리오 문서 |
| `app/app.py` | `FIRST_MESSAGE` (첫 인사), `PRESET_QUESTIONS` (추천 질문), `KEYWORDS` (통계 키워드) |

**순서**: 1) `data/candidate_profile.py` 수정 → 2) `data/portfolio/` 문서 교체 → 3) `uv run python scripts/build_index.py` → 4) 앱 재시작

---

## 📁 프로젝트 구조

```
scy_Rag/
├── app.py                      # Gradio 진입점 (HF Spaces / 로컬)
├── main.py                     # 패키지 플레이스홀더
├── config.py                   # 전체 설정 파라미터
├── CLAUDE.md                   # Claude Code 프로젝트 가이드
│
├── app/
│   ├── app.py                  # Gradio UI (채팅, 통계, 요약 다운로드)
│   ├── rag.py                  # RAG 파이프라인 (라우터, 검색, 생성)
│   ├── rag_eval.py             # Faithfulness / Relevance 평가
│   └── CLAUDE.md
│
├── scripts/
│   ├── build_index.py          # 문서 → FAISS + BM25 인덱스 빌드
│   ├── evaluate_rag.py         # RAG 평가 스크립트
│   └── CLAUDE.md
│
├── data/
│   ├── portfolio/              # PDF·DOCX 포트폴리오 문서
│   ├── candidate_profile.py    # 후보자 프로필 & 프로젝트 목록
│   └── CLAUDE.md
│
├── index/                      # FAISS + BM25 인덱스 (빌드 후 생성)
├── docs/                       # 아키텍처 문서, 다이어그램
├── notebooks/                  # 실험용 노트북
│
├── pyproject.toml              # uv 의존성 (Python >=3.13)
├── requirements.txt            # HF Spaces용 pip 의존성
├── .env.example                # 환경 변수 템플릿
└── .github/workflows/          # GitHub Actions (HF Space 동기화)
```

---

## 📊 RAG 평가

```bash
uv run python scripts/evaluate_rag.py
```

샘플 쿼리 3개로 Faithfulness / Relevance 점수를 출력합니다. 인덱스 빌드가 선행되어야 합니다.

---

## 🌐 구글 드라이브 사용

1. [Google Cloud Console](https://console.cloud.google.com/)에서 **Google Drive API** 사용 설정
2. OAuth 2.0 데스크톱 앱 클라이언트로 `credentials.json` 발급
3. `~/.credentials/credentials.json`에 배치 (또는 `.env`의 `GOOGLE_DRIVE_CREDENTIALS_PATH`로 지정)
4. 폴더 URL의 ID를 `.env`의 `GOOGLE_DRIVE_FOLDER_ID`에 설정
5. `uv run python scripts/build_index.py` 실행

지원 형식: Google Docs, Google 스프레드시트, PDF

---

## 🔬 LangSmith 트레이싱

`.env`에 아래를 설정하면 LangChain 실행이 [LangSmith](https://smith.langchain.com/)에 기록됩니다:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=scy-rag    # 선택
```

---

## 🚢 배포 (Hugging Face Spaces)

### Space 생성

1. [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. SDK: **Gradio**, Visibility: **Private** 권장

### Space에 포함할 파일

| 포함 | 경로 | 설명 |
|:----:|------|------|
| ✅ | `app.py`, `config.py`, `app/`, `data/` | 핵심 코드 |
| ✅ | `index/` | FAISS + BM25 (로컬 빌드 후 커밋 또는 HF Dataset) |
| ✅ | `requirements.txt` | pip 의존성 |
| ❌ | `.env` | **Settings → Secrets**로 등록 |
| ❌ | `token.json`, `credentials.json` | 로컬 전용 |

### Secrets 설정

Space → **Settings → Variables and secrets** → `OPENAI_API_KEY` 등록

### HF Dataset으로 인덱스 제공

대용량 인덱스 파일은 [HF Dataset](https://huggingface.co/datasets)에 업로드 후:
- Space Variable: `HF_INDEX_DATASET=본인아이디/데이터셋이름`
- Private 접근 시: `HF_TOKEN`을 Secrets에 추가

### GitHub → HF 자동 동기화

1. [HF 설정](https://huggingface.co/settings/tokens)에서 **Write** 토큰 발급
2. GitHub → **Settings → Secrets → Actions** → `HF_TOKEN` 등록
3. `git push origin main` 시 `.github/workflows/`가 Space로 자동 미러링

---

## 📚 관련 문서

- [코드 흐름·함수 역할](docs/코드_동작_설명.md)
- [RAG 파이프라인 구조 (Mermaid)](docs/RAG_ARCHITECTURE.md)
- [아키텍처 다이어그램 (HTML)](docs/architecture_diagram.html)
- [질문 흐름 다이어그램 (HTML)](docs/question_flow_diagram.html)
