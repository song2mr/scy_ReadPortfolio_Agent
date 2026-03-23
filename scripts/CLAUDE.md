# scripts/ — 인덱스 빌드 & 평가

## 인덱스 빌드

```bash
uv run python scripts/build_index.py
```

- `data/portfolio/`의 PDF·DOCX를 청킹 후 FAISS + BM25(Kiwi) 인덱스 생성
- 출력: `index/index.faiss`, `index/index.pkl`, `index/bm25_corpus.pkl`, `index/bm25_docs.pkl`
- Google Drive 연동: `.env`에 `GOOGLE_DRIVE_FOLDER_ID` 설정 시 드라이브에서 로드

## RAG 평가

```bash
uv run python scripts/evaluate_rag.py
```

- 샘플 쿼리 3개로 Faithfulness / Relevance 점수 출력
- 인덱스 빌드 선행 필요
