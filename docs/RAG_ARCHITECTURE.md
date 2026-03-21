# RAG 전체 구조 (Mermaid)

## 1. 요청~응답 흐름 (flowchart)

```mermaid
flowchart TB
    subgraph input["입력"]
        Q["질문 (question)"]
        H["대화 히스토리 (history_pairs)"]
    end

    subgraph router["라우터 (LLM)"]
        ROUTE["_route_question()"]
        ROUTE --> RAG["RAG"]
        ROUTE --> NO_RAG["NO_RAG"]
        ROUTE --> GENERAL["GENERAL"]
    end

    Q --> ROUTE
    H --> ROUTE

    subgraph no_rag_path["NO_RAG 경로"]
        NO_LLM["LLM (NO_RAG_PORTFOLIO_SYSTEM)"]
        NO_RAG --> NO_LLM
    end

    subgraph general_path["GENERAL 경로"]
        GEN_LLM["LLM (GENERAL_SYSTEM)"]
        GENERAL --> GEN_LLM
    end

    subgraph rag_path["RAG 경로"]
        direction TB
        IDX["인덱스 로드\n(FAISS + BM25)"]
        QE["Query Expansion"]
        HYBRID["Hybrid Search\n(쿼리별)"]
        RRF["RRF 병합"]
        RERANK["Reranker"]
        RAG_LLM["LLM (RAG_SYSTEM + context)"]
        EVAL["평가·재시도 (선택)"]

        RAG --> IDX
        IDX --> QE
        QE --> HYBRID
        HYBRID --> RRF
        RRF --> RERANK
        RERANK --> RAG_LLM
        RAG_LLM --> EVAL
    end

    NO_LLM --> OUT["답변 + source_docs"]
    GEN_LLM --> OUT
    EVAL --> OUT
```

## 2. RAG 경로 상세 (Query Expansion → Hybrid → RRF → Reranker)

- **Query Expansion**: 1차 검색 없이 **질문만** LLM에 넣어 재작성. 단일 → 1줄, 포괄적 → 2~4개 하위 질문.
- **쿼리당 문서 수**: 각 쿼리마다 Hybrid Search(Dense + Sparse) 후 RRF로 **쿼리당 `HYBRID_MERGE_TOP_N`개**(예: 15개) 문서를 가져옴. 그 다음 여러 쿼리의 랭킹을 다시 RRF로 병합해 최종 15개 → Reranker → 4개.

```mermaid
flowchart LR
    subgraph expansion["Query Expansion"]
        direction TB
        LLM_Q["LLM 재작성\n(질문만, Few-shot)"]
        QLIST["쿼리 리스트\n[원본, 하위1, 하위2, ...]"]
        LLM_Q --> QLIST
    end

    subgraph per_query["쿼리별 Hybrid Search (쿼리당 HYBRID_MERGE_TOP_N개)"]
        direction TB
        D["Dense (FAISS)\ntop-k"]
        S["Sparse (BM25/Kiwi)\ntop-k"]
        RRF1["RRF 병합\n→ 쿼리당 1개 랭킹\n(각 랭킹 N개 문서)"]
        D --> RRF1
        S --> RRF1
    end

    subgraph merge["다중 랭킹 병합"]
        RRF2["RRF 병합\n(모든 쿼리 결과)"]
        TOP["상위 N개"]
        RRF2 --> TOP
    end

    subgraph final["최종 정제"]
        RERANK["Reranker\n(Cross-Encoder)"]
        TOP_N["상위 4개 → context"]
        RERANK --> TOP_N
    end

    QLIST --> per_query
    per_query --> RRF2
    TOP --> RERANK
    TOP_N --> GEN["LLM 답변 생성"]
```

## 3. 데이터/인덱스 구조

```mermaid
flowchart TB
    subgraph build["인덱스 빌드 (scripts/build_index.py)"]
        DOCS["PDF/DOCX\n(data/portfolio 또는 Drive)"]
        CHUNK["청킹\n(RecursiveCharacterTextSplitter)"]
        CHUNKS["청크 리스트"]
        DOCS --> CHUNK
        CHUNK --> CHUNKS

        subgraph index_dir["index/ (HF 업로드 시 전체)"]
            FAISS["index.faiss\nindex.pkl"]
            BM25_C["bm25_corpus.pkl"]
            BM25_D["bm25_docs.pkl"]
        end

        CHUNKS --> FAISS
        CHUNKS --> BM25_C
        CHUNKS --> BM25_D
    end

    subgraph runtime["런타임 로드 (app/rag.py)"]
        VS["Vectorstore (FAISS)"]
        B["BM25 + Kiwi\n(없으면 Dense만)"]
        FAISS -.-> VS
        BM25_C -.-> B
        BM25_D -.-> B
    end
```

## 4. 컴포넌트 요약

| 구분 | 내용 |
|------|------|
| **라우터** | LLM 1회 호출 → RAG / NO_RAG / GENERAL (대화 히스토리 참고) |
| **Query Expansion** | 1차 검색 없음. 질문만으로 LLM 재작성(Few-shot). 단일 → 1줄, 포괄적 → 2~4개 하위 질문 |
| **쿼리당 문서 수** | 각 쿼리마다 Hybrid Search 후 RRF로 **HYBRID_MERGE_TOP_N개**(예: 15개). 이 랭킹들을 다시 RRF 병합 |
| **Hybrid (쿼리당)** | Dense(FAISS) top-k + Sparse(BM25/Kiwi) top-k → RRF로 하나의 랭킹 |
| **RRF** | 여러 쿼리의 랭킹을 RRF로 병합 → 상위 HYBRID_MERGE_TOP_N개 |
| **Reranker** | Cross-Encoder(bge-reranker-v2-m3) → 상위 RERANKER_TOP_N(4)개만 LLM에 전달 |
| **평가·재시도** | Faithfulness/Relevance 낮으면 k 늘려 1회 재검색·재생성 |
