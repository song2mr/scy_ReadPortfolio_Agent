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
- 포트폴리오 데이터·출처 YAML: `data/CLAUDE.md`
- 전체 설정 파라미터: `config.py`
- RAG 소스 확장 스펙: `docs/RAG_SOURCE_EXPANSION_SPEC.md`
- 체인 흐름(키·메타): `docs/chain_flow.html` (브라우저)

## 설계 한 줄

요약으로 파일을 고르고 → **`rag_scope`** 로 “목록(요약만, 기본은 요약 전체)” vs “한 프로젝트(본문 전체)”를 가른다. 개인/회사 표기는 **`data/portfolio_origins.yaml`** → 빌드 시 청크 메타 → 답변 컨텍스트에 라벨.
