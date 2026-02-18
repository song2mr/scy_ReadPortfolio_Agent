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
RERANKER_TOP_N = 4                             # 재순위화 후 LLM에 전달할 문서 수
RETRIEVE_K_INITIAL = 15                        # Reranker 사용 시 초기 검색 k (넓게 가져옴)

# 임베딩
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# LLM
OPENAI_MODEL = "gpt-5-mini"           # 답변 생성(RAG·NO_RAG·GENERAL)용
OPENAI_MODEL_ROUTER = "gpt-4o-mini"   # 라우터(RAG/NO_RAG/GENERAL 분류)용. 가벼운 모델로 두면 비용·지연 감소
