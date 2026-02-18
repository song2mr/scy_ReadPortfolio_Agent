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

PDF/Word 포트폴리오를 올려두면, 인사팀이 채팅으로 질문할 수 있는 RAG 에이전트입니다.  
Gradio + LangChain + FAISS + OpenAI. 배포는 Hugging Face Spaces(Private 권장).

## 기능 요약

- **3-way 라우터**: 질문을 한 번에 RAG / NO_RAG(인사·메타) / GENERAL(무관)으로 분류. 대화 히스토리를 참고해 맥락 있는 분류.
- **RAG 답변**: 포트폴리오 문서 검색(FAISS) 후 OpenAI로 답변 생성. 참고 문단 접기 표시.
- **스트리밍**: 답변이 토큰 단위로 실시간 출력.
- **대화 요약 다운로드**: MD / PDF(한글 실패 시 TXT)로 대화 요약 + 전체 내역 저장.
- **질문 키워드 통계**: 경력, 강점, 프로젝트 등 키워드 언급 횟수 표시.
- **평가 후 재시도**: Faithfulness/Relevance 점수가 낮으면 k를 늘려 1회 재검색·재생성 (config로 on/off).

## 빠른 시작

### 1. 환경

- Python 3.11+ (uv 권장)
- `.env` 에 `OPENAI_API_KEY=sk-...` 설정

```bash
# 의존성
uv sync

# .env 설정
cp .env.example .env
# .env 에 OPENAI_API_KEY 입력
```

### 2. 포트폴리오 넣기

- **로컬**: `data/portfolio/` 에 PDF 또는 Word(.docx) 파일을 넣습니다.
- **구글 드라이브**: `.env` 에 `GOOGLE_DRIVE_FOLDER_ID` 를 설정하면, 해당 폴더의 Google Docs·스프레드시트·PDF를 사용합니다. (아래 [구글 드라이브 사용](#구글-드라이브-사용) 참고)

### 3. 인덱스 빌드

```bash
uv run python scripts/build_index.py
```

- `index/` 에 FAISS 인덱스가 생성됩니다.
- 문서를 바꾼 뒤에는 이 명령을 다시 실행하세요.
- 구글 드라이브를 쓰는 경우 `.env` 에 `GOOGLE_DRIVE_FOLDER_ID` 가 있으면 드라이브에서 로드하고, 없으면 `data/portfolio/` 에서 로드합니다.

### 4. 앱 실행

**프로젝트 루트**에서 다음 중 하나로 실행합니다.

```bash
uv run python app.py
```

또는

```bash
uv run python -m app.app
```

브라우저에서 열리는 주소로 접속해 채팅, 추천 질문, 대화 초기화, 요약 다운로드(MD/PDF)를 사용하면 됩니다.

## 설정 (config.py)

| 항목 | 설명 | 기본값 |
|------|------|--------|
| `OPENAI_MODEL` | 답변 생성(RAG·NO_RAG·GENERAL)용 모델 | `gpt-5-mini` |
| `OPENAI_MODEL_ROUTER` | 라우터(RAG/NO_RAG/GENERAL 분류)용 모델 | `gpt-4o-mini` |
| `RETRIEVE_K` | RAG 검색 시 가져올 청크 수 | `6` |
| `EVAL_RETRY_ENABLED` | 평가 후 재시도 사용 여부 | `True` |
| `EVAL_MIN_FAITHFULNESS` / `EVAL_MIN_RELEVANCE` | 이 점수 미만이면 k 늘려 재시도 | `3` |
| `RETRIEVE_K_RETRY` | 재시도 시 k | `8` |
| `EMBEDDING_MODEL` | FAISS 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 인덱스 빌드 시 청킹 | `800` / `150` |

라우터는 분류만 하므로 `OPENAI_MODEL_ROUTER`를 더 가벼운 모델로 두면 비용·지연을 줄일 수 있습니다.

## 구글 드라이브 사용

데이터를 로컬이 아닌 **구글 드라이브 폴더**에 두고, 그 폴더를 청킹·임베딩에 사용할 수 있습니다.

1. **Google Cloud Console**  
   - [Google Cloud Console](https://console.cloud.google.com/) 에서 프로젝트 생성 후 **Google Drive API** 사용 설정.
   - **API 및 서비스 → 사용자 인증 정보**에서 **데스크톱 앱**용 OAuth 2.0 클라이언트 ID 생성 후 `credentials.json` 다운로드.

2. **인증 파일 배치**  
   - `credentials.json` 을 `~/.credentials/credentials.json` 에 넣거나, `.env` 에 `GOOGLE_DRIVE_CREDENTIALS_PATH` 로 경로 지정.  
   - 첫 실행 시 브라우저에서 구글 로그인/동의 후 `token.json` 이 생성됩니다.

3. **.env 설정**  
   - 구글 드라이브에서 사용할 **폴더**를 열고, URL `https://drive.google.com/drive/folders/xxxx` 에서 `xxxx` 부분이 폴더 ID입니다.  
   - `.env` 에 예: `GOOGLE_DRIVE_FOLDER_ID=xxxx`  
   - (선택) `GOOGLE_DRIVE_CREDENTIALS_PATH`, `GOOGLE_DRIVE_TOKEN_PATH`, `GOOGLE_DRIVE_RECURSIVE=true` 등 설정.

4. **인덱스 빌드**  
   - `uv run python scripts/build_index.py` 실행 시 `GOOGLE_DRIVE_FOLDER_ID` 가 있으면 해당 폴더에서 문서를 가져와 청킹·임베딩 후 `index/` 에 FAISS를 저장합니다.  
   - 지원 형식: Google Docs, Google 스프레드시트, PDF.

## 디버깅

앱이 무한 로딩되거나 RAG 동작을 확인하고 싶을 때: `.env` 에 `DEBUG_RAG=1` 을 넣고 터미널에서 앱을 실행하면, 라우터 분류·검색·스트리밍 단계별로 `[RAG]`, `[APP]` 로그가 출력됩니다.

## 지원자 맞춤화

첫 인사 문구·추천 질문·키워드 통계용 키워드는 `app/app.py` 의 `FIRST_MESSAGE`, `PRESET_QUESTIONS`, `KEYWORDS` 에서 수정할 수 있습니다.  
RAG/NO_RAG 시스템 프롬프트와 지원자 기본 정보(경력·활동 요약)는 `app/rag.py` 의 `PROFILE_BASIC`, `RAG_SYSTEM`, `NO_RAG_PORTFOLIO_SYSTEM`, `GENERAL_SYSTEM` 에서 수정할 수 있습니다.

## 대화 히스토리(맥락) 반영 방식

LLM이 이전 대화를 기억하고 이어서 답하도록, **매 질문마다 최근 대화를 프롬프트에 함께 넣는 방식**을 씁니다. 별도 메모리 DB나 세션 저장 없이, 호출할 때마다 “지금까지의 대화 중 최근 N턴”을 전달합니다.

1. **UI에서 백엔드로 전달**  
   Gradio Chatbot에 쌓인 전체 대화(질문·답변 쌍)가 사용자가 전송할 때마다 `history_pairs` 형태로 `get_answer` / `get_answer_stream` 에 넘어갑니다.

2. **라우터(분류)에서의 맥락**  
   질문을 RAG / NO_RAG / GENERAL 로 나누기 위해, **최근 5턴**을 턴당 최대 200자로 잘라 문자열로 만든 뒤, 라우터 전용 프롬프트의 `<recent_conversation>` 안에 넣습니다. 그래서 “아까 경력 이야기했는데, 그럼 프로젝트는?”처럼 짧은 후속 질문도 맥락을 보고 RAG로 분류할 수 있습니다.

3. **답변 생성에서의 맥락**  
   RAG·NO_RAG·GENERAL 모두, **최근 5턴**을 LangChain의 `HumanMessage` / `AIMessage` 리스트로 바꾼 뒤, 프롬프트의 `MessagesPlaceholder("chat_history")` 자리에 넣습니다. 따라서 LLM에는  
   `[시스템 지시] + [최근 5턴의 사용자/봇 메시지] + [현재 질문]`  
   이 한 번에 전달되고, 그 안에 포함된 이전 질문·답변을 보고 이어서 답합니다.

4. **요약**  
   - 전체 대화는 UI가 보관하고, 매 요청 시 **최근 5턴**만 추려서 라우터와 답변 생성에 사용합니다.  
   - 토큰·비용을 줄이기 위해 “최근 N턴” 슬라이딩 윈도우 방식이며, N은 `app/rag.py` 의 `_pairs_to_messages(max_turns=5)`, `_format_history_for_router(max_turns=5)` 에서 조정할 수 있습니다.

## 프로젝트 구조

```
├── app.py              # Gradio 진입점 (Hugging Face Spaces / 로컬)
├── config.py           # 경로, 청킹, 검색, LLM 모델 등 설정
├── app/
│   ├── app.py          # UI 구성 (build_ui), 전송·요약·초기화 이벤트
│   ├── rag.py          # 라우터, RAG 체인, get_answer / get_answer_stream
│   └── rag_eval.py     # Faithfulness / Relevance 평가 (evaluate_response_from_docs)
├── scripts/
│   ├── build_index.py  # 로컬 data/portfolio 또는 구글 드라이브 → 청킹 → 임베딩 → index/ FAISS 저장
│   └── evaluate_rag.py # RAG 평가 스크립트 (선택)
├── data/portfolio/     # PDF·DOCX 원본
├── index/              # FAISS 인덱스 (build_index.py 실행 후 생성)
├── docs/
│   └── 코드_동작_설명.md
└── .env                # OPENAI_API_KEY (git 제외)
```

## RAG 평가 (선택)

쿼리·참고 문단·답변 품질(Faithfulness / Relevance)을 LLM으로 평가할 때:

```bash
uv run python scripts/evaluate_rag.py
```

- 인덱스가 있어야 함. 샘플 쿼리로 답변 생성 후 1~5점·이유 출력.
- 자체 쿼리/로그로 평가하려면 `app/rag_eval.py` 의 `evaluate_response` / `evaluate_response_from_docs` 를 호출하면 됩니다.

## 배포 (Hugging Face Spaces)

무료 계정으로 **Private** Space를 만들어 다른 사람이 보지 못하게 배포할 수 있습니다. (무료 private 저장 100GB 이내, CPU Basic 하드웨어 무료.)

### 1. Space 만들기

1. [huggingface.co](https://huggingface.co) 에서 로그인(또는 회원가입).
2. [Spaces 페이지](https://huggingface.co/spaces) → **Create new Space**.
3. 설정:
   - **Name**: 원하는 Space 이름 (예: `portfolio-rag`).
   - **SDK**: **Gradio** 선택.
   - **Visibility**: **Private** 선택 (다른 사용자는 404만 보임).
   - (선택) License: MIT 등.
4. 생성 후 빈 Space 저장소가 만들어집니다.

### 2. Space에 넣을 파일

로컬에서 아래 구조대로 필요한 파일만 복사해 Space repo에 푸시합니다. **`.env`는 넣지 않고**, API 키는 Space 설정의 Secrets로 넣습니다.

| 포함 여부 | 경로 | 설명 |
|----------|------|------|
| ✅ | `app.py` | 루트 진입점 (Gradio `demo` 정의). Space가 이 파일을 실행합니다. |
| ✅ | `config.py` | 경로, 청킹, 모델 등 설정 |
| ✅ | `app/` | `app.py`, `rag.py`, `rag_eval.py` 전체 폴더 |
| ✅ | `scripts/` | `build_index.py` (인덱스 없을 때 빌드용, 선택) |
| ✅ | `index/` | FAISS 인덱스 (`*.faiss`, `*.pkl` 등). **미리 로컬에서 빌드한 뒤** 커밋해서 올리세요. |
| ✅ | `data/portfolio/` | 배포 시 사용할 PDF·DOCX (선택). 인덱스를 repo에 넣으면 없어도 됨. |
| ✅ | `requirements.txt` | Space용 의존성 (pip 설치). 프로젝트 루트에 있음. |
| ❌ | `.env` | 절대 푸시 금지. API 키는 Space **Settings → Secrets** 에 등록. |
| ❌ | `token.json`, `credentials.json` | 구글 드라이브 인증 파일 (로컬 전용). |

- **인덱스**: 로컬에서 `uv run python scripts/build_index.py` 로 `index/` 를 만든 뒤, `index/` 폴더 통째로 Space repo에 포함하면 됩니다. 인덱스를 넣지 않으면 앱 시작 시 문서가 없어 RAG가 동작하지 않을 수 있습니다.
- **진입점**: Space는 기본적으로 루트의 `app.py` 를 실행합니다. 이 파일이 `app.app.build_ui()` 로 UI를 만들고 `demo` 를 내놓으므로 별도 설정 없이 동작합니다.

### 3. Secrets 설정

1. Space 저장소 페이지 → **Settings** → **Variables and secrets**.
2. **New secret** 추가:
   - 이름: `OPENAI_API_KEY`
   - 값: `sk-...` (OpenAI API 키).
3. (선택) 구글 드라이브를 Space에서 쓰려면 `GOOGLE_DRIVE_FOLDER_ID`, `GOOGLE_DRIVE_CREDENTIALS_PATH` 등 필요한 변수/시크릿을 같은 화면에서 추가. (일반적으로 로컬에서만 쓰고 Space에는 인덱스만 올리는 방식을 권장.)

### 4. 푸시 후 확인

- `git add` → `git commit` → `git push` 하면 Space가 자동으로 빌드·실행됩니다.
- 무료 CPU Basic은 사용하지 않으면 **슬립**됩니다. 링크로 다시 접속하면 깨우는 데 시간이 걸릴 수 있습니다.
- Private이면 Space URL을 아는 사람만 접속할 수 없고, 로그인한 본인만 볼 수 있습니다. (다른 사용자는 404.)

### 5. 요약

| 항목 | 내용 |
|------|------|
| Private 여부 | Space 생성/설정에서 **Visibility: Private** 선택 가능, 무료. |
| 비용 | 저장 100GB·CPU Basic 무료. 추가 하드웨어는 유료. |
| 진입점 | 루트 `app.py` (Gradio `demo`). |
| API 키 | 코드에 넣지 말고 Space **Settings → Secrets** 에 `OPENAI_API_KEY` 등록. |
| 인덱스 | 로컬에서 빌드한 `index/` 를 repo에 포함해 푸시. |

---

자세한 코드 흐름·함수 역할은 `docs/코드_동작_설명.md` 를 참고하세요.
