"""프로젝트 설정 (MEMO.md 5-2 기술 상세 반영)."""
import os
from pathlib import Path

# UI 표시용 앱 버전 (Gradio 제목·푸터). Space Variables에서 APP_VERSION 으로 덮어쓸 수 있음.
APP_VERSION = (os.getenv("APP_VERSION") or "1.11").strip() or "1.11"

# 경로 (프로젝트 루트 기준)
ROOT = Path(__file__).resolve().parent
PORTFOLIO_DIR = ROOT / "data" / "portfolio"
PORTFOLIO_ORIGINS_PATH = ROOT / "data" / "portfolio_origins.yaml"
INDEX_DIR = ROOT / "index"

# Google Drive (선택): 설정 시 빌드 시 구글 드라이브 폴더에서 문서 로드
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
# 인덱스 빌드 문서 소스: 기본 True(로컬 data/portfolio만). 드라이브 빌드는 GOOGLE_DRIVE_FOLDER_ID 넣고 False로.
INDEX_BUILD_USE_LOCAL_ONLY = os.getenv("INDEX_BUILD_USE_LOCAL_ONLY", "true").lower() in ("true", "1", "yes")
GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "").strip()
GOOGLE_DRIVE_TOKEN_PATH = os.getenv("GOOGLE_DRIVE_TOKEN_PATH", "").strip()
GOOGLE_DRIVE_RECURSIVE = os.getenv("GOOGLE_DRIVE_RECURSIVE", "false").lower() in ("true", "1", "yes")

# 청킹 (마크다운 ##/### 단위 유지하려면 separators는 build_index에서 설정)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 인덱스 빌드 시 파일(source)별 LLM 요약 청크 추가 (metadata: chunk_kind=summary). OPENAI_API_KEY 필요.
INDEX_SUMMARY_ENABLED = os.getenv("INDEX_SUMMARY_ENABLED", "true").lower() in ("true", "1", "yes")
INDEX_SUMMARY_MODEL = os.getenv("INDEX_SUMMARY_MODEL", "").strip() or "gpt-4o-mini"
INDEX_SUMMARY_MAX_INPUT_CHARS = int(os.getenv("INDEX_SUMMARY_MAX_INPUT_CHARS", "14000"))

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

# RAG 범위: 요약 청크로 문서(source) 선택 → SINGLE=한 파일 본문 전체, BROAD=요약만 여러 개
RAG_SUMMARY_ROUTING_ENABLED = True
# FAISS에서 가져온 뒤 요약만 남길 때 초기 후보 수(부족하면 이전만으로도 동작)
RAG_SUMMARY_DENSE_POOL = 40
RAG_BROAD_MAX_SUMMARY_SOURCES = 6   # BROAD + RAG_BROAD_USE_ALL_SUMMARIES=False 일 때만: Rerank 기반 최대 파일 수
RAG_BROAD_MAX_SUMMARY_CHUNKS = 12    # BROAD + 전체 요약 끔일 때만: 요약 청크 상한
# BROAD면 인덱스 요약 전체(소스당 1청크)를 컨텍스트에 넣음 — 목록·회상형 질문 안전망 (끄면 아래 max만큼 Rerank 기반)
RAG_BROAD_USE_ALL_SUMMARIES = True
# "어떤 프로젝트 해봤지?" 등 나열·회상형 질문을 LLM 전에 BROAD로 고정 (SINGLE 오분류 완화)
RAG_SCOPE_BROAD_HEURISTIC = True

# Rerank 이후 동일 source(파일) 청크 확장 — 프로젝트당 한 파일 구조에 맞춰 뜨문뜨문 검색을 완화
SOURCE_EXPANSION_ENABLED = True
# 확장할 source 최대 개수(Rerank 결과에서 처음 등장한 파일 순). 초과 파일은 확장하지 않음
SOURCE_EXPANSION_MAX_SOURCES = 6
# 파일당 청크 상한. 0이면 문자 상한만 적용
SOURCE_EXPANSION_MAX_CHUNKS_PER_SOURCE = 0
# 확장 후 컨텍스트 문자 합 상한(구분선 포함 전체 _format_docs 길이에 가깝게 제한)
SOURCE_EXPANSION_MAX_CONTEXT_CHARS = 48_000

# 임베딩
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# LLM
OPENAI_MODEL = "gpt-5"         # 답변 생성(RAG·NO_RAG·GENERAL)용 (실제 사용 모델명으로 설정. 예: gpt-4o)
OPENAI_MODEL_ROUTER = "gpt-4o-mini"   # 라우터(RAG/NO_RAG/GENERAL 분류)용. 가벼운 모델로 두면 비용·지연 감소

# 대화 히스토리
MAX_HISTORY_TURNS = 5      # LLM에 전달하는 최근 대화 최대 턴 수
RETRY_K_INCREMENT = 5      # 평가 재시도 시 dense_k / sparse_k 증가량
