# scripts/ — 인덱스 빌드 & 평가

## 인덱스 빌드

```bash
uv run python scripts/build_index.py
```

- `data/portfolio/`의 PDF·DOCX·**Markdown(.md)** 를 청킹 후 FAISS + BM25(Kiwi) 인덱스 생성
- 파일당 **요약 청크**(`chunk_kind=summary`)는 `OPENAI_API_KEY`가 있을 때 LLM으로 생성·인덱스에 포함(없으면 본문만). `INDEX_SUMMARY_ENABLED=false`로 끌 수 있음(`config.py`/환경변수)
- 출력: `index/index.faiss`, `index/index.pkl`, `index/bm25_corpus.pkl`, `index/bm25_docs.pkl`
- 문서 소스: 기본 **로컬** `data/portfolio/`. 드라이브는 `GOOGLE_DRIVE_FOLDER_ID` + `INDEX_BUILD_USE_LOCAL_ONLY=false`
- 출처 유형: `data/portfolio_origins.yaml` — `personal`/`company`/`unspecified`/`excluded`. `excluded` 및 미매칭+`default: excluded` 는 인덱스 미포함
- 점검·yaml 갱신: `uv run python scripts/portfolio_origins_audit.py` (미매칭 파일에 규칙 자동 추가; `--dry-run`은 쓰기 없음). `default: excluded` 면 자동 추가 안 함 → 넣을 파일만 수동 `rules` 에 추가

## RAG 평가

```bash
uv run python scripts/evaluate_rag.py
```

- 샘플 쿼리 3개로 Faithfulness / Relevance 점수 출력
- 인덱스 빌드 선행 필요
