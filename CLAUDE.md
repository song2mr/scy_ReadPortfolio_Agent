# scy_Rag

포트폴리오 문서(PDF/DOCX)를 RAG로 검색해 인사 담당자 질문에 답하는 Gradio 챗봇.

## 빠른 시작

```bash
cp .env.example .env            # OPENAI_API_KEY 설정
uv run python scripts/build_index.py  # 인덱스 생성
uv run python app.py                  # 앱 실행
```

## 핵심 환경 변수

- `OPENAI_API_KEY` — 필수
- `APP_PASSWORD` — 미설정 시 "dev"로 입장
- `HF_INDEX_DATASET` — HF Spaces 배포 시 인덱스 다운로드 경로
- `DEBUG_RAG=1` — 터미널 디버그 로그
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` — 트레이스 활성화

## 상세 문서

- RAG 파이프라인 & UI: `app/CLAUDE.md`
- 인덱스 빌드 & 평가: `scripts/CLAUDE.md`
- 포트폴리오 데이터 관리: `data/CLAUDE.md`
- 전체 설정 파라미터: `config.py`
