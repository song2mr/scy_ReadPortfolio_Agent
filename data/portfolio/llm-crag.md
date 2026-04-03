---
source_repo: https://github.com/song2mr/llm-crag
sync_note: GitHub README 미러 (자동 수집). 원본과 다를 수 있음.
---

**GitHub 저장소:** https://github.com/song2mr/llm-crag

# agent_portfolio

LangChain / LangGraph 기반 AI 에이전트 포트폴리오 프로젝트입니다.

## 프로젝트 구조

```
agent_portfolio/
├── .env                  # API 키 및 환경변수 (text-to-sql에서 복사)
├── pyproject.toml        # uv 의존성 정의
├── uv.lock               # 의존성 잠금 파일 (uv sync 후 생성)
├── main.py               # 엔트리포인트
├── src/
│   └── agent_portfolio/
│       └── __init__.py   # 패키지 루트
├── portfolio_chatbot/    # 기존 서브프로젝트
└── scy_Rag/              # 기존 서브프로젝트
```

## 의존성 설치

```bash
# uv가 설치되어 있어야 합니다
uv sync
```

## 실행

```bash
uv run python main.py
```

## 주요 패키지

| 패키지 | 용도 |
|---|---|
| `langchain`, `langchain-openai` | LLM 체인 구성 |
| `langgraph` | 에이전트 그래프 워크플로우 |
| `langsmith` | 트레이싱 및 디버깅 |
| `faiss-cpu` | 벡터 유사도 검색 |
| `sentence-transformers` | 임베딩 모델 |
| `gradio` | 웹 UI |
| `python-dotenv` | 환경변수 관리 |

## 환경변수 (.env)

```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=...
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```
