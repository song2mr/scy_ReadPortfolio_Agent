"""프로젝트 설정 (MEMO.md 5-2 기술 상세 반영)."""
import os
from pathlib import Path

# 경로 (프로젝트 루트 기준)
ROOT = Path(__file__).resolve().parent
PORTFOLIO_DIR = ROOT / "data" / "portfolio"
INDEX_DIR = ROOT / "index"

# Google Drive (선택): 설정 시 빌드 시 구글 드라이브 폴더에서 문서 로드
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "").strip()
GOOGLE_DRIVE_TOKEN_PATH = os.getenv("GOOGLE_DRIVE_TOKEN_PATH", "").strip()
GOOGLE_DRIVE_RECURSIVE = os.getenv("GOOGLE_DRIVE_RECURSIVE", "false").lower() in ("true", "1", "yes")

# 청킹 (마크다운 ##/### 단위 유지하려면 separators는 build_index에서 설정)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 검색 (k 늘리면 더 많은 문단 후보 → 답 품질·포함률 향상)
RETRIEVE_K = 6
# 평가 후 재시도: Faithfulness/Relevance 낮으면 k 늘려 재검색·재생성 (1회)
EVAL_RETRY_ENABLED = True
EVAL_MIN_FAITHFULNESS = 3  # 이 점수 미만이면 재시도
EVAL_MIN_RELEVANCE = 3
RETRIEVE_K_RETRY = 8       # 재시도 시 가져올 청크 수 (k 증가)

# Reranker (Cross-Encoder): 검색 후 정밀 재순위화
RERANKER_ENABLED = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # 다국어(한국어 포함) Cross-Encoder
RERANKER_TOP_N = 8                             # 재순위화 후 LLM에 전달할 문서 수
RETRIEVE_K_INITIAL = 15                        # Reranker 사용 시 초기 검색 k (넓게 가져옴)

# Hybrid Search (Dense + BM25, RRF 병합)
HYBRID_DENSE_K = 7         # Dense(FAISS) per-query top-k
HYBRID_SPARSE_K = 7         # Sparse(BM25) per-query top-k
HYBRID_MERGE_TOP_N = 15      # RRF 병합 후 Reranker 전 상위 개수
RRF_K = 60                   # RRF 상수 (score = 1/(k+rank))

# Query Expansion (질문만으로 LLM 재작성, 1차 검색 없음. 포괄적 질문은 여러 하위 질문으로 분할)
QUERY_EXPANSION_MAX_SUB_QUERIES = 5  # 포괄적 질문 시 하위 질문 최대 개수 (각각 RAG 검색 후 RRF 병합)

# 임베딩
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# LLM
OPENAI_MODEL = "gpt-5"         # 답변 생성(RAG·NO_RAG·GENERAL)용 (실제 사용 모델명으로 설정. 예: gpt-4o)
OPENAI_MODEL_ROUTER = "gpt-4o-mini"   # 라우터(RAG/NO_RAG/GENERAL 분류)용. 가벼운 모델로 두면 비용·지연 감소

# 대화 히스토리
MAX_HISTORY_TURNS = 5      # LLM에 전달하는 최근 대화 최대 턴 수
RETRY_K_INCREMENT = 5      # 평가 재시도 시 dense_k / sparse_k 증가량
