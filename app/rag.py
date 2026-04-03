"""
RAG 체인: 포폴 관련 여부 라우팅, Query Expansion + Hybrid Search(FAISS+BM25/Kiwi) + RRF → Reranker → OpenAI 답변.
HF 업로드 시 index/ (index.faiss, index.pkl, bm25_corpus.pkl, bm25_docs.pkl) 포함하면 동일 동작.
"""
import os
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 (app/ 기준 상위)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# config는 루트에 있음
import sys
sys.path.insert(0, str(ROOT))
import config

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from app.rag_eval import build_evaluation_chain
from data.candidate_profile import PROFILE_BASIC, QUERY_EXPANSION_TOPICS

# LangSmith: RAG 단계(라우터, Query Expansion, Hybrid Search, Rerank 등)를 트레이스에 단계별로 보이게 함
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def _deco(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]  # @traceable (without parens)
        return _deco


# 인덱스 metadata.portfolio_origin → LLM에 보이는 한글 라벨
_PORTFOLIO_ORIGIN_LABELS_KO = {
    "personal": "개인 프로젝트",
    "company": "회사·기관 소속 업무 또는 산출물",
    "unspecified": "출처 유형 미표시 (문서만 참고)",
}


def _portfolio_origin_label(metadata: dict) -> str:
    o = (metadata or {}).get("portfolio_origin")
    if not o and (metadata or {}).get("personal_project"):
        o = "personal"
    return _PORTFOLIO_ORIGIN_LABELS_KO.get(o or "unspecified", _PORTFOLIO_ORIGIN_LABELS_KO["unspecified"])


def _format_docs(docs):
    """컨텍스트에 출처 유형을 붙여 LLM이 개인/회사 구분을 따를 수 있게 함."""
    parts = []
    for doc in docs:
        md = getattr(doc, "metadata", None) or {}
        label = _portfolio_origin_label(md)
        src = (md.get("source") or md.get("title") or "").strip()
        try:
            fn = Path(src).name if src else ""
        except Exception:
            fn = ""
        header = f"[참고 문서 출처 유형: {label}]"
        if fn:
            header += f" (파일·문서명: {fn})"
        parts.append(f"{header}\n{(doc.page_content or '')}")
    return "\n\n---\n\n".join(parts)


def _strip_html(text) -> str:
    """Gradio 등에서 bot 이 리스트로 올 수 있음 → 문자열로 통일 후 처리."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _pairs_to_messages(pairs: list, max_turns: int = config.MAX_HISTORY_TURNS) -> list:
    """대화 pairs [[user,bot],...] → LangChain 메시지 리스트 (최근 max_turns턴만)."""
    if not pairs:
        return []
    out = []
    for user, bot in pairs[-max_turns:]:
        if user is not None and str(user).strip():
            out.append(HumanMessage(content=user))
        if bot is not None and str(bot).strip():
            out.append(AIMessage(content=_strip_html(bot)))
    return out


def _debug(msg: str) -> None:
    """DEBUG_RAG=1 일 때만 터미널에 출력 (앱 무한 로딩 디버깅용)."""
    if os.getenv("DEBUG_RAG", "").strip() in ("1", "true", "yes"):
        print(f"[RAG] {msg}", flush=True)


# 다운로드 후 index.faiss가 하위 폴더에 있으면 그 경로 사용 (HF Dataset 구조 대응)
_resolved_index_dir: Path | None = None


def _find_index_dir(base: Path) -> Path | None:
    """base 또는 그 하위에서 index.faiss 위치 반환."""
    if not base.exists():
        return None
    if (base / "index.faiss").exists():
        return base
    for sub in base.iterdir():
        if sub.is_dir() and (sub / "index.faiss").exists():
            return sub
    return None


def _ensure_index_dir():
    """로컬에 인덱스가 없고 HF_INDEX_DATASET 이 설정되어 있으면 Dataset/Model에서 index/ 로 다운로드."""
    global _resolved_index_dir
    index_dir = Path(config.INDEX_DIR)
    found = _find_index_dir(index_dir)
    if found:
        _resolved_index_dir = found
        return
    hf_dataset = os.getenv("HF_INDEX_DATASET", "").strip()
    if not hf_dataset:
        print("[RAG] HF_INDEX_DATASET 미설정. Space 설정 → Variables에 HF_INDEX_DATASET=아이디/데이터셋이름 추가 후 재시작하세요.", flush=True)
        return
    try:
        from huggingface_hub import snapshot_download
        print(f"[RAG] HF에서 인덱스 다운로드 중: {hf_dataset}", flush=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        token = os.getenv("HF_TOKEN", "").strip() or None
        for repo_type in ("dataset", "model"):
            try:
                snapshot_download(
                    repo_id=hf_dataset,
                    repo_type=repo_type,
                    local_dir=str(index_dir),
                    token=token,
                )
                found = _find_index_dir(index_dir)
                if found:
                    _resolved_index_dir = found
                    print(f"[RAG] 인덱스 다운로드 완료. ({found})", flush=True)
                    return
            except Exception as e:
                if repo_type == "dataset":
                    print(f"[RAG] Dataset 다운로드 실패, Model 시도: {e}", flush=True)
                else:
                    raise
        print("[RAG] index.faiss를 찾을 수 없습니다. Dataset/Model 루트에 index.faiss, index.pkl을 올렸는지 확인하세요.", flush=True)
    except Exception as e:
        print(f"[RAG] HF 인덱스 다운로드 실패: {e}", flush=True)


def _load_retriever(k: int | None = None):
    """k 미지정 시 Reranker 활성이면 RETRIEVE_K_INITIAL, 아니면 RETRIEVE_K 사용."""
    if k is not None:
        use_k = k
    elif getattr(config, "RERANKER_ENABLED", False):
        use_k = getattr(config, "RETRIEVE_K_INITIAL", 15)
    else:
        use_k = config.RETRIEVE_K
    _ensure_index_dir()
    load_dir = _resolved_index_dir if _resolved_index_dir is not None else Path(config.INDEX_DIR)
    if not (load_dir / "index.faiss").exists():
        raise FileNotFoundError(f"index.faiss 없음: {load_dir}. HF_INDEX_DATASET 설정·다운로드 확인하세요.")
    _debug("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    _debug("FAISS 인덱스 로드 중...")
    vectorstore = FAISS.load_local(
        str(load_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _debug("retriever 준비 완료")
    return vectorstore.as_retriever(search_kwargs={"k": use_k})


# ── Vectorstore / BM25 싱글톤 (Hybrid Search, Query Expansion) ─────
_vectorstore = None
_bm25_tuple = None  # (bm25, docs_list, tokenize_fn) or None
_corpus_chunks_list = None  # bm25_docs.pkl 전체 청크 (source 확장용). None=미로드, []=파일 없음
_bm25_summary_built = False
_bm25_summary_tuple = None  # (bm25, summary_docs_only, tokenize) or None — 요약 청크만 BM25


def _get_vectorstore():
    """FAISS 벡터스토어 로드 (캐시). HF에서 index/ 다운로드 후 동일 경로에서 로드."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    _ensure_index_dir()
    load_dir = _resolved_index_dir if _resolved_index_dir is not None else Path(config.INDEX_DIR)
    if not (load_dir / "index.faiss").exists():
        raise FileNotFoundError(f"index.faiss 없음: {load_dir}")
    _debug("임베딩·FAISS 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    _vectorstore = FAISS.load_local(
        str(load_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _debug("vectorstore 준비 완료")
    return _vectorstore


def _get_bm25():
    """BM25 + Kiwi 토크나이저 로드 (캐시). 파일 없으면 None → Dense만 사용."""
    global _bm25_tuple
    if _bm25_tuple is not None:
        return _bm25_tuple
    _ensure_index_dir()
    load_dir = _resolved_index_dir if _resolved_index_dir is not None else Path(config.INDEX_DIR)
    corpus_path = load_dir / "bm25_corpus.pkl"
    docs_path = load_dir / "bm25_docs.pkl"
    if not corpus_path.exists() or not docs_path.exists():
        _debug("BM25 인덱스 없음 → Dense만 사용")
        return None
    _debug("BM25(Kiwi) 로드 중...")
    with open(corpus_path, "rb") as f:
        corpus_tokens = pickle.load(f)
    with open(docs_path, "rb") as f:
        bm25_docs = pickle.load(f)
    if len(bm25_docs) != len(corpus_tokens):
        _debug("BM25 corpus와 docs 길이 불일치 → Dense만 사용")
        return None
    bm25 = BM25Okapi(corpus_tokens)
    from kiwipiepy import Kiwi
    _kiwi = Kiwi()

    def tokenize(text: str):
        return [t.form for t in _kiwi.tokenize(text)]

    _bm25_tuple = (bm25, bm25_docs, tokenize)
    _debug("BM25 준비 완료")
    return _bm25_tuple


def _get_bm25_summary_tuple():
    """요약 청크(chunk_kind=summary)만 모은 BM25·문서 리스트. 구 인덱스(요약 없음)면 None."""
    global _bm25_summary_built, _bm25_summary_tuple
    if _bm25_summary_built:
        return _bm25_summary_tuple
    _bm25_summary_built = True
    all_chunks = _get_corpus_chunks_list()
    if not all_chunks:
        _bm25_summary_tuple = None
        return None
    summ_docs = [d for d in all_chunks if _chunk_is_summary(d)]
    if not summ_docs:
        _bm25_summary_tuple = None
        _debug("요약 청크 없음 → 요약 라우팅 생략 가능")
        return None
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    corpus_tokens = []
    for doc in summ_docs:
        tokens = [t.form for t in kiwi.tokenize(doc.page_content or "")]
        corpus_tokens.append(tokens)
    bm25_s = BM25Okapi(corpus_tokens)

    def tokenize(text: str):
        return [t.form for t in kiwi.tokenize(text)]

    _bm25_summary_tuple = (bm25_s, summ_docs, tokenize)
    _debug(f"BM25(요약 전용): {len(summ_docs)}개 청크")
    return _bm25_summary_tuple


def _get_corpus_chunks_list() -> list | None:
    """인덱스에 저장된 전체 청크 리스트(빌드 순서). source 확장 전용; BM25 비활성이어도 pkl만 있으면 로드."""
    global _corpus_chunks_list
    if _corpus_chunks_list is not None:
        return _corpus_chunks_list if len(_corpus_chunks_list) > 0 else None
    _ensure_index_dir()
    load_dir = _resolved_index_dir if _resolved_index_dir is not None else Path(config.INDEX_DIR)
    docs_path = load_dir / "bm25_docs.pkl"
    if not docs_path.exists():
        _corpus_chunks_list = []
        return None
    with open(docs_path, "rb") as f:
        _corpus_chunks_list = pickle.load(f)
    return _corpus_chunks_list if _corpus_chunks_list else None


def _doc_source_key(doc) -> str:
    md = getattr(doc, "metadata", None) or {}
    return (md.get("source") or md.get("title") or "").strip() or "__missing__"


def _chunk_is_summary(doc) -> bool:
    """인덱스의 요약 청크는 본문 확장 시 제외(중복·토큰 절감). 메타 없으면 구 인덱스로 간주해 본문 취급."""
    kind = (getattr(doc, "metadata", None) or {}).get("chunk_kind")
    return kind == "summary"


def _all_summary_docs_one_per_source(all_chunks: list) -> list:
    """빌드·저장 순서 유지, 각 source당 요약 청크 1개 — BROAD 시 전체 개요 레이어."""
    seen: set[str] = set()
    out: list = []
    for d in all_chunks:
        if not _chunk_is_summary(d):
            continue
        sk = _doc_source_key(d)
        if sk in seen:
            continue
        seen.add(sk)
        out.append(d)
    return out


@traceable(name="source_expand", run_type="chain")
def _expand_same_source_after_rerank(reranked: list) -> list:
    """Rerank 상위 청크에 나온 파일(source)별로, 인덱스의 동일 파일 청크를 문서 순으로 합침."""
    if not getattr(config, "SOURCE_EXPANSION_ENABLED", False) or not reranked:
        return reranked
    all_chunks = _get_corpus_chunks_list()
    if not all_chunks:
        return reranked

    sources_order = []
    seen = set()
    for d in reranked:
        s = _doc_source_key(d)
        if s not in seen:
            seen.add(s)
            sources_order.append(s)
    max_src = getattr(config, "SOURCE_EXPANSION_MAX_SOURCES", 0)
    if max_src and len(sources_order) > max_src:
        sources_order = sources_order[:max_src]

    wanted = set(sources_order)
    by_source = {s: [] for s in sources_order}
    for d in all_chunks:
        if _chunk_is_summary(d):
            continue
        s = _doc_source_key(d)
        if s in wanted:
            by_source[s].append(d)

    max_per = getattr(config, "SOURCE_EXPANSION_MAX_CHUNKS_PER_SOURCE", 0)
    expanded = []
    for s in sources_order:
        group = by_source.get(s) or []
        if max_per and len(group) > max_per:
            group = group[:max_per]
        expanded.extend(group)

    if not expanded:
        return reranked

    max_chars = getattr(config, "SOURCE_EXPANSION_MAX_CONTEXT_CHARS", 0)
    sep = "\n\n---\n\n"
    if not max_chars or max_chars <= 0:
        _debug(f"Source 확장: Rerank {len(reranked)}개 → 동일 source 전체 {len(expanded)}개 청크")
        return expanded

    out: list = []
    total = 0
    for d in expanded:
        text = d.page_content or ""
        extra = len(sep) if out else 0
        need = len(text) + extra
        if not out and len(text) > max_chars:
            out.append(
                Document(
                    page_content=text[:max_chars],
                    metadata=dict(getattr(d, "metadata", None) or {}),
                )
            )
            break
        if total + need > max_chars:
            break
        out.append(d)
        total += need
    if not out:
        return reranked
    _debug(
        f"Source 확장: Rerank {len(reranked)}개 → {len(out)}개 청크 "
        f"(≤{max_chars}자, sources≤{len(sources_order)})"
    )
    return out


# Query Expansion: 질문만으로 LLM 재작성 (1차 검색 없음. 단일/포괄적 질문 → 1개 또는 여러 하위 쿼리)
# 포트폴리오·경험 조회 시 쿼리 재작성에 참고할 프로젝트·솔루션 목록은 data/candidate_profile.py 에 정의

REFORMULATION_TEMPLATE = """<role>
당신은 지원자 포트폴리오 문서 검색을 위한 쿼리 재작성 전문가입니다. 사용자 질문을 검색에 유리하도록 재작성합니다. 동의어·구체적 키워드·관련 개념을 넣어도 됩니다.
</role>

<portfolio_topics>
아래는 지원자의 실제 프로젝트·경험 목록입니다. 포트폴리오나 경험 조회가 필요한 질문일 때 재작성 쿼리에 구체적인 프로젝트명·키워드를 활용하세요.
{portfolio_topics}
</portfolio_topics>

<constraint>
위 포트폴리오·경험 목록에 있는 항목만 쿼리에 구체 키워드로 넣으세요. 목록에 없는 프로젝트나 경험은 지어내지 마세요.
</constraint>

<rules>
1. 단일·구체적 질문이면: 재작성된 질문을 한 줄로만 출력하세요.
2. 포괄적 질문(경력+강점+프로젝트 등 여러 주제를 한 번에 묻는 경우)이면: 2~4개의 하위 질문으로 나누어, 각각 한 줄씩 출력하세요. (줄마다 하나의 검색 쿼리가 됩니다.)
</rules>

<output_format>
[재작성된 질문] 아래에, 한 줄에 하나의 질문만. 여러 개면 줄바꿈으로 구분.
</output_format>

<examples>
예시 1 (단일)
[원본 질문] 이 사람 뭐 해?
[재작성된 질문] 이 지원자의 주요 경력, 현재 직무와 담당 업무, 강점을 알려주세요.

예시 2 (단일)
[원본 질문] 프로젝트 알려줘
[재작성된 질문] 데이터 분석·연구 관련 프로젝트, 학회 활동, 논문 경험과 사용 기술을 구체적으로 알려주세요.

예시 3 (포괄적 → 여러 하위 질문)
[원본 질문] 경력이랑 강점이랑 프로젝트 다 알려줘
[재작성된 질문]
이 지원자의 주요 경력과 현재 직무, 담당 업무는 무엇인가요?
이 지원자의 강점과 역량을 알려주세요.
데이터 분석·연구 관련 프로젝트, 학회 활동, 논문 경험을 구체적으로 알려주세요.
</examples>

<actual_task>
[원본 질문]
{question}

[재작성된 질문]
</actual_task>
"""


@traceable(name="query_expansion", run_type="chain")
def _expand_query(question: str) -> list:
    """질문만으로 LLM 재작성(단일 또는 여러 하위 질문) → [원본, 재작성1, 재작성2, ...] 반환. 실패 시 [원본]만. (1차 검색 없음)"""
    question = (question or "").strip()
    if not question:
        return [""]
    max_sub_queries = getattr(config, "QUERY_EXPANSION_MAX_SUB_QUERIES", 5)
    try:
        llm = ChatOpenAI(
            model=getattr(config, "OPENAI_MODEL_ROUTER", config.OPENAI_MODEL),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        prompt = _safe_format(
            REFORMULATION_TEMPLATE,
            question=question[:500],
            portfolio_topics=QUERY_EXPANSION_TOPICS.strip(),
        )
        out = llm.invoke(prompt).content.strip()
        if not out:
            return [question]
        # 한 줄에 하나의 질문. 번호/불릿 제거 후 비어있지 않은 줄만 수집 (순서 유지)
        lines = []
        for line in out.replace("\r", "\n").split("\n"):
            line = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            line = re.sub(r"^[-*]\s*", "", line).strip()
            if line and len(line) > 2:
                lines.append(line)
        # 중복 제거(순서 유지), 원본과 동일한 것 제거, 상한
        seen = set()
        reformulated = []
        for q in lines:
            q_lower = q.lower().strip()
            if q_lower != question.lower().strip() and q_lower not in seen:
                seen.add(q_lower)
                reformulated.append(q)
                if len(reformulated) >= max_sub_queries:
                    break
        if not reformulated:
            return [question]
        _debug(f"Query Expansion: 원본 + 재작성 {len(reformulated)}개")
        return [question, *reformulated]
    except Exception as e:
        _debug(f"Query Expansion 실패, 원본만 사용: {e}")
    return [question]


@traceable(name="rrf_merge")
def _rrf_merge_multiple(rankings: list, top_n: int = 15) -> list:
    """여러 랭킹을 RRF로 병합, page_content 기준 중복 제거 후 상위 top_n개. RRF 상수는 config.RRF_K 사용."""
    rrf_k = getattr(config, "RRF_K", 60)
    scores = {}
    for rank_list in rankings:
        for rank, doc in enumerate(rank_list):
            key = (doc.page_content or "")[:400]
            if key not in scores:
                scores[key] = (doc, 0.0)
            scores[key] = (scores[key][0], scores[key][1] + 1.0 / (rrf_k + rank + 1))
    sorted_docs = sorted(scores.values(), key=lambda x: -x[1])
    return [doc for doc, _ in sorted_docs[:top_n]]


@traceable(name="hybrid_retrieve", run_type="retriever")
def _hybrid_retrieve(query: str, vectorstore, bm25_tuple, dense_k: int, sparse_k: int, merge_top_n: int) -> list:
    """Dense + BM25 검색 후 RRF 병합. BM25 없으면 Dense만 merge_top_n개."""
    dense_docs = vectorstore.similarity_search(query, k=dense_k)
    if bm25_tuple is None:
        return dense_docs[:merge_top_n]
    bm25, bm25_docs, tokenize = bm25_tuple
    q_tokens = tokenize(query)
    if not q_tokens:
        return dense_docs[:merge_top_n]
    scores = bm25.get_scores(q_tokens)
    if len(scores) == 0:
        return dense_docs[:merge_top_n]
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:sparse_k]
    sparse_docs = [bm25_docs[i] for i in top_indices]
    return _rrf_merge_multiple([dense_docs, sparse_docs], top_n=merge_top_n)


@traceable(name="hybrid_retrieve_summaries", run_type="retriever")
def _hybrid_retrieve_summaries(
    query: str,
    vectorstore,
    bm25_summ_tuple,
    dense_k: int,
    sparse_k: int,
    merge_top_n: int,
) -> list:
    """요약 청크에 대해서만 Hybrid: Dense는 풀에서 요약만 필터, Sparse는 요약 전용 BM25."""
    pool = max(dense_k * 3, getattr(config, "RAG_SUMMARY_DENSE_POOL", 40))
    raw_dense = vectorstore.similarity_search(query, k=pool)
    dense_docs = [d for d in raw_dense if _chunk_is_summary(d)][:dense_k]
    if bm25_summ_tuple is None:
        return dense_docs[:merge_top_n]
    bm25, summ_docs, tokenize = bm25_summ_tuple
    q_tokens = tokenize(query)
    if not q_tokens:
        return dense_docs[:merge_top_n]
    scores = bm25.get_scores(q_tokens)
    if len(scores) == 0:
        return dense_docs[:merge_top_n]
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:sparse_k]
    sparse_docs = [summ_docs[i] for i in top_indices]
    return _rrf_merge_multiple([dense_docs, sparse_docs], top_n=merge_top_n)


def _apply_char_budget_docs(docs: list, max_chars: int) -> list:
    """문서 리스트를 문자 상한까지 앞에서부터 포함(구분선 길이 반영)."""
    if not max_chars or max_chars <= 0 or not docs:
        return docs
    sep = "\n\n---\n\n"
    out: list = []
    total = 0
    for d in docs:
        text = d.page_content or ""
        extra = len(sep) if out else 0
        need = len(text) + extra
        if not out and len(text) > max_chars:
            out.append(
                Document(
                    page_content=text[:max_chars],
                    metadata=dict(getattr(d, "metadata", None) or {}),
                )
            )
            break
        if total + need > max_chars:
            break
        out.append(d)
        total += need
    return out


# ── Reranker (Cross-Encoder) ────────────────────────────────────────
_reranker: CrossEncoder | None = None


def _load_reranker() -> CrossEncoder | None:
    """Reranker 모델 싱글톤 로드. 비활성이면 None 반환."""
    global _reranker
    if not getattr(config, "RERANKER_ENABLED", False):
        return None
    if _reranker is not None:
        return _reranker
    model_name = getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    _debug(f"Reranker 모델 로드 중... ({model_name})")
    _reranker = CrossEncoder(model_name)
    _debug("Reranker 로드 완료")
    return _reranker


@traceable(name="rerank", run_type="chain")
def _rerank_docs(query: str, docs: list, top_n: int | None = None) -> list:
    """Cross-Encoder로 문서를 재순위화하여 상위 top_n개만 반환.
    Reranker 비활성이거나 문서가 적으면 원본 그대로 반환."""
    reranker = _load_reranker()
    if reranker is None or not docs:
        return docs
    top_n = top_n or getattr(config, "RERANKER_TOP_N", 4)
    if len(docs) <= top_n:
        return docs
    pairs = [(query, doc.page_content or "") for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored_docs[:top_n]]
    _debug(f"Rerank 완료: {len(docs)}개 → {len(reranked)}개 (scores: {sorted(scores, reverse=True)[:5]})")
    return reranked


# 라우터: RAG 필요 포폴 / RAG 불필요 포폴 / 일반 — 한 번에 구분 (대화 히스토리 참고)
ROUTER_PROMPT = """<task>
현재 질문을 아래 세 가지 중 하나로만 분류하세요. 대화 맥락이 있으면 반드시 참고하세요.
</task>

<criteria>
- RAG: 지원자 포트폴리오(이력서·경력·프로젝트·학력 등) 문서를 검색해서 구체적 내용으로 답해야 하는 질문. (예: 주요 프로젝트?, 강점?, 어디 근무? / 이전에 경력 이야기 중이면 "프로젝트는?"도 RAG)
- NO_RAG: 포트폴리오 Q&A 맥락이지만 문서 검색 없이 답할 수 있는 질문. (예: 안녕, 너 누구야?, 뭐 할 수 있어?)
- GENERAL: 포트폴리오와 무관한 질문. (예: 날씨, 일반 상식, 인사 이야기)
</criteria>

<edge_case>
이전 답변에서 이미 말한 내용을 다시 물어보는 경우(예: "그거 자세히", "방금 말한 프로젝트", "그 프로젝트 더 알려줘")는, 문서를 다시 검색할 필요 없이 대화 맥락만으로 이어서 답할 수 있으면 NO_RAG로 분류하세요.
</edge_case>

<recent_conversation>
{history_text}
</recent_conversation>

<current_question>
{question}
</current_question>

<output>
반드시 RAG, NO_RAG, GENERAL 중 한 단어만 출력하세요.
</output>"""


def _parse_route_output(raw: str) -> str:
    """라우터 LLM 출력 문자열 → RAG | NO_RAG | GENERAL."""
    r = (raw or "").strip().upper()
    if "NO_RAG" in r or "NO RAG" in r:
        return "NO_RAG"
    if "GENERAL" in r:
        return "GENERAL"
    return "RAG"


def _format_history_for_router(pairs: list, max_turns: int = config.MAX_HISTORY_TURNS, max_chars_per_turn: int = 200) -> str:
    """라우터용으로 최근 대화만 짧게 문자열화. 없으면 '(없음)'."""
    if not pairs:
        return "(없음)"
    lines = []
    for user, bot in pairs[-max_turns:]:
        u = _strip_html(user or "")[:max_chars_per_turn]
        b = _strip_html(bot or "")[:max_chars_per_turn]
        if u:
            lines.append(f"사용자: {u}")
        if b:
            lines.append(f"봇: {b}")
    return "\n".join(lines) if lines else "(없음)"


def _safe_format(template: str, **kwargs: str) -> str:
    """format 시 사용자 입력에 있는 중괄호를 이스케이프하여 치환 오류/주입 방지."""
    escaped = {k: (v or "").replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    return template.format(**escaped)


RAG_SCOPE_PROMPT = """<task>
포트폴리오 RAG로 답할 때, 질문이 **여러 문서·프로젝트를 목록·개요 위주로 묻는지(BROAD)**,
**특정 한 프로젝트·한 주제를 깊게 묻는지(SINGLE)** 만 분류하세요.
대화 맥락이 있으면 참고하세요.
</task>

<criteria>
- BROAD: 둘 이상 또는 전반·비교·목록·나열·회상·경험 훑기. (예: "LLM 관련 프로젝트 알려줘", "어떤 프로젝트 있어?", "어떤 프로젝트 해봤지?", "뭐 만들어봤어?", "뭘 했어?", "경험 정리해줘", "포트폴리오에 뭐 있어?", "주요 프로젝트", "기술 스택 비교")
- SINGLE: **이미 특정한 하나**의 프로젝트·제품·문서·주제를 가리키며 깊게 묻는 경우. (예: "ETF 추천봇 자세히", "보드게임 RAG 어떻게 만들었어?", "그 프로젝트만 설명", "SageMaker 예산 모델 구조는?", "애드메이커 솔루션 스택은?")
- 주의: **"어떤/무슨/뭐" + 프로젝트·경험·해봤**처럼 범위가 넓은 짧은 질문은 거의 항상 BROAD. 특정 고유명사·프로젝트 제목 없이 전체를 묻는 톤이면 BROAD.
</criteria>

<recent_conversation>
{history_text}
</recent_conversation>

<current_question>
{question}
</current_question>

<output>
반드시 BROAD 또는 SINGLE 한 단어만 출력하세요.
</output>"""


def _parse_rag_scope(raw: str) -> str:
    r = (raw or "").strip().upper()
    if "BROAD" in r:
        return "BROAD"
    return "SINGLE"


def _rag_scope_heuristic_broad(question: str) -> bool:
    """LLM 오분류 보조: 나열·회상형 질문은 BROAD로 두지 않으면 한 파일 본문만 들어가 취약."""
    q = (question or "").strip().lower()
    if not q:
        return False
    patterns = (
        r"어떤\s*프로젝트",
        r"무슨\s*프로젝트",
        r"프로젝트\s*(?:뭐|뭘|어떤|무엇|있)",
        r"(?:뭐|뭘)\s*(?:해봤|해\s*봤|했어|만들었|만들어\s*봤)",
        r"해본\s*(?:거|것|프로젝트|경험|일)",
        r"해봤\s*지",
        r"경험\s*(?:뭐|어떤|있|정리)",
        r"(?:주요|대표)\s*프로젝트",
        r"프로젝트\s*(?:목록|리스트|정리|소개|알려)",
        r"어떤\s*(?:거|것)\s*해봤",
        r"뭘\s*했",
        r"뭔\s*프로젝트",
        r"프로젝트\s*경험",
    )
    return any(re.search(p, q) for p in patterns)


def _classify_rag_scope(question: str, history_pairs: list | None = None) -> str:
    """BROAD=요약만 여러 파일, SINGLE=요약으로 고른 1파일 본문 전부 (고유 소스 개수와 무관)."""
    if not getattr(config, "RAG_SUMMARY_ROUTING_ENABLED", True):
        return "SINGLE"
    if getattr(config, "RAG_SCOPE_BROAD_HEURISTIC", True) and _rag_scope_heuristic_broad(question):
        _debug("rag_scope 휴리스틱 → BROAD (나열/회상형)")
        return "BROAD"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "SINGLE"
    history_pairs = history_pairs or []
    history_text = _format_history_for_router(history_pairs)
    try:
        llm = ChatOpenAI(
            model=getattr(config, "OPENAI_MODEL_ROUTER", config.OPENAI_MODEL),
            temperature=0,
            api_key=api_key,
        )
        prompt = _safe_format(
            RAG_SCOPE_PROMPT,
            question=question[:500],
            history_text=history_text[:2000],
        )
        out = llm.invoke(prompt).content
        return _parse_rag_scope(out)
    except Exception as e:
        _debug(f"rag_scope 분류 실패, SINGLE: {e}")
        return "SINGLE"


def _rag_scope_step(state: dict) -> dict:
    """LCEL: 질문·history_pairs → rag_scope(BROAD|SINGLE)."""
    q = (state.get("question") or "").strip()
    pairs = state.get("history_pairs") or []
    scope = _classify_rag_scope(q, pairs)
    _debug(f"RAG scope: {scope}")
    return {**state, "rag_scope": scope}


def _build_router_chain():
    """LCEL: router_prompt | llm | StrOutputParser | parse_route. LangSmith에 라우터 단계·입출력 노출."""
    llm = ChatOpenAI(
        model=getattr(config, "OPENAI_MODEL_ROUTER", config.OPENAI_MODEL),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([("human", ROUTER_PROMPT)])
    return (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_route_output, name="parse_route")
    ).with_config(run_name="router")


def _build_query_expansion_chain():
    """LCEL: expansion_prompt | llm | StrOutputParser | parse_queries. 입력 question → 출력 {question, chat_history, queries}."""
    llm = ChatOpenAI(
        model=getattr(config, "OPENAI_MODEL_ROUTER", config.OPENAI_MODEL),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([("human", REFORMULATION_TEMPLATE)]).partial(
        portfolio_topics=QUERY_EXPANSION_TOPICS.strip()
    )

    def _parse_queries(state: dict) -> dict:
        question = (state.get("question") or "").strip()
        text = (state.get("expansion_text") or "").strip()
        max_n = getattr(config, "QUERY_EXPANSION_MAX_SUB_QUERIES", 5)
        if not text:
            return {**state, "queries": [question]}
        lines = []
        for line in text.replace("\r", "\n").split("\n"):
            line = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            line = re.sub(r"^[-*]\s*", "", line).strip()
            if line and len(line) > 2:
                lines.append(line)
        seen = set()
        reformulated = []
        for q in lines:
            q_lower = q.lower().strip()
            if q_lower != question.lower() and q_lower not in seen:
                seen.add(q_lower)
                reformulated.append(q)
                if len(reformulated) >= max_n:
                    break
        queries = [question, *reformulated] if reformulated else [question]
        return {**state, "queries": queries}

    expansion_step = prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(expansion_text=expansion_step)
        | RunnableLambda(_parse_queries, name="parse_queries")
    ).with_config(run_name="query_expansion")
    return chain


def _retrieve_legacy_hybrid_expand(
    question: str,
    queries: list,
    vs,
    bm25,
    dense_k: int,
    sparse_k: int,
    merge_top_n: int,
) -> tuple[list, str, list]:
    """구 방식: 전체 청크 hybrid → Rerank → 동일 source 본문 확장."""
    rankings = [_hybrid_retrieve(q, vs, bm25, dense_k, sparse_k, merge_top_n) for q in queries]
    merged = _rrf_merge_multiple(rankings, top_n=merge_top_n)
    source_docs = _expand_same_source_after_rerank(_rerank_docs(question, merged))
    context = _format_docs(source_docs)
    preview = [(doc.page_content or "")[:300] for doc in source_docs]
    return source_docs, context, preview


def _run_retrieval_core(
    question: str,
    queries: list,
    rag_scope: str = "SINGLE",
    dense_k: int | None = None,
    sparse_k: int | None = None,
    merge_top_n: int | None = None,
) -> tuple[list, str, list]:
    """요약으로 source 후보 정한 뒤, rag_scope=BROAD면 요약만 여러 파일·SINGLE이면 1순위 source 본문 전부. 요약 인덱스 없으면 legacy."""
    dense_k = dense_k if dense_k is not None else getattr(config, "HYBRID_DENSE_K", 10)
    sparse_k = sparse_k if sparse_k is not None else getattr(config, "HYBRID_SPARSE_K", 10)
    merge_top_n = merge_top_n if merge_top_n is not None else getattr(config, "HYBRID_MERGE_TOP_N", 15)
    vs = _get_vectorstore()
    bm25_full = _get_bm25()

    if not getattr(config, "RAG_SUMMARY_ROUTING_ENABLED", True):
        return _retrieve_legacy_hybrid_expand(
            question, queries, vs, bm25_full, dense_k, sparse_k, merge_top_n
        )

    all_chunks = _get_corpus_chunks_list()
    bm25_summ = _get_bm25_summary_tuple()
    if not all_chunks or bm25_summ is None:
        return _retrieve_legacy_hybrid_expand(
            question, queries, vs, bm25_full, dense_k, sparse_k, merge_top_n
        )

    rankings = [
        _hybrid_retrieve_summaries(q, vs, bm25_summ, dense_k, sparse_k, merge_top_n)
        for q in queries
    ]
    merged = _rrf_merge_multiple(rankings, top_n=merge_top_n)
    top_summaries = _rerank_docs(question, merged)

    sources_order = []
    seen = set()
    for d in top_summaries:
        sk = _doc_source_key(d)
        if sk not in seen:
            seen.add(sk)
            sources_order.append(sk)

    if not sources_order:
        return _retrieve_legacy_hybrid_expand(
            question, queries, vs, bm25_full, dense_k, sparse_k, merge_top_n
        )

    max_chars = getattr(config, "SOURCE_EXPANSION_MAX_CONTEXT_CHARS", 0)
    scope = (rag_scope or "SINGLE").upper()
    if scope not in ("SINGLE", "BROAD"):
        scope = "SINGLE"
    multi_source = scope == "BROAD"
    _debug(
        f"RAG retrieval: rag_scope={scope} (sources in summary rank: {len(sources_order)}) "
        f"→ {'summaries only' if multi_source else 'full body (top-1 source)'}"
    )

    if multi_source:
        if getattr(config, "RAG_BROAD_USE_ALL_SUMMARIES", True):
            source_docs = _all_summary_docs_one_per_source(all_chunks)
            _debug(f"BROAD: 전체 요약 레이어 source {len(source_docs)}개 (Rerank는 SINGLE용 1순위만 사용)")
        else:
            max_src = getattr(config, "RAG_BROAD_MAX_SUMMARY_SOURCES", 6)
            max_chunks = getattr(config, "RAG_BROAD_MAX_SUMMARY_CHUNKS", 12)
            source_docs = []
            picked_src = set()
            for d in top_summaries:
                if len(source_docs) >= max_chunks:
                    break
                sk = _doc_source_key(d)
                if sk in picked_src:
                    continue
                if len(picked_src) >= max_src:
                    break
                picked_src.add(sk)
                source_docs.append(d)
        if max_chars and max_chars > 0:
            source_docs = _apply_char_budget_docs(source_docs, max_chars)
    else:
        target = sources_order[0]
        body_chunks = [
            d for d in all_chunks
            if _doc_source_key(d) == target and not _chunk_is_summary(d)
        ]
        if not body_chunks:
            return _retrieve_legacy_hybrid_expand(
                question, queries, vs, bm25_full, dense_k, sparse_k, merge_top_n
            )
        if max_chars and max_chars > 0:
            source_docs = _apply_char_budget_docs(body_chunks, max_chars)
        else:
            source_docs = body_chunks

    context = _format_docs(source_docs)
    preview = [(doc.page_content or "")[:300] for doc in source_docs]
    return source_docs, context, preview


def _retrieve_step(state: dict) -> dict:
    """LCEL 내부: question + queries + rag_scope → 검색 후 context/source_docs."""
    question = state.get("question") or ""
    queries = state.get("queries") or [question]
    rag_scope = state.get("rag_scope") or "SINGLE"
    try:
        source_docs, context, retrieved_preview = _run_retrieval_core(
            question, queries, rag_scope
        )
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        return {**state, "context": "", "source_docs": [], "retrieved_preview": []}
    return {
        **state,
        "context": context,
        "source_docs": source_docs,
        "retrieved_preview": retrieved_preview,
    }


def _build_retrieve_chain():
    """LCEL: RunnableLambda(retrieve_step). 입력 {question, queries} → 출력에 context, source_docs, retrieved_preview 포함."""
    return RunnableLambda(_retrieve_step, name="retrieve").with_config(run_name="retrieve")


def _build_rag_retrieve_pipeline():
    """Query Expansion → rag_scope → Hybrid 검색·리랭크·컨텍스트 조립. LangSmith: `rag_retrieve_pipeline`."""
    return (
        _build_query_expansion_chain()
        | RunnableLambda(_rag_scope_step, name="rag_scope_step")
        | _build_retrieve_chain()
    ).with_config(run_name="rag_retrieve_pipeline")


def _build_rag_generate_chain():
    """LCEL: RAG prompt | llm | StrOutputParser. 입력 {question, chat_history, context, sub_queries} → answer."""
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    return (prompt | llm | StrOutputParser()).with_config(run_name="generate")


def _route_question(question: str, history_pairs: list | None = None) -> str:
    """질문을 RAG / NO_RAG / GENERAL 로 분류 (평가·스크립트 호환). 실제 앱은 LCEL `router_assign` 사용."""
    st = _build_entry_runnable().invoke({"question": question or "", "history_pairs": history_pairs or []})
    if not st.get("question"):
        return "NO_RAG"
    st2 = _build_router_assign_runnable().invoke(st)
    return st2.get("route") or "RAG"


# 지원자 기본 정보는 data/candidate_profile.py 에서 PROFILE_BASIC 으로 임포트

# 시스템 프롬프트: 송찬영을 인사 담당자에게 소개하는 에이전트 (송찬영 = 데이터 분석가·LLM 에이전트 개발·설계자)
RAG_SYSTEM = """<role>당신은 송찬영을 인사·채용 담당자에게 소개하는 에이전트입니다. 지원자 송찬영은 데이터 분석가이자 LLM 에이전트 개발·설계자입니다. 당신은 그 포지션의 어시스턴트가 아니라, 인사 담당자가 질문하면 송찬영의 포트폴리오(이력서·경력기술서·프로젝트)를 RAG로 검색해 답변하는 소개용 에이전트입니다.</role>

<basic_info>
송찬영 | 이엠넷 R&D본부 재직 중 (2024.01~), 에코마케팅 인턴 (2023.01~04)
한국외대 광고PR브랜딩/통계학과 졸업 (GPA 3.92/4.5), KCI 1저자 논문 보유

[LLM/Agent 설계]
- Copilot Studio + Power Automate + SageMaker 연계 광고 데이터 분석 Agent 설계 → 리포트 시간 80% 단축
- 사내 문서 RAG 기반 Q&A Agent: 규칙 기반 → 오케스트레이션 구조 전환, Evaluation 단계 직접 설계 → 문의 해결 시간 70% 단축
- 개인 프로젝트로 LangChain 기반 Modular RAG 구현 (Hybrid Search + Reranker + Faithfulness/Relevance 평가 + 재시도 로직)
- 핵심 인사이트: "챗봇 vs 자동화" 문제 유형별 구조 분리가 LLM 서비스 설계의 핵심

[자동화 솔루션]
- Meta Marketing API + C# 기반 광고 자동 등록 시스템 (비공식 Canvas API 우회 포함) → 등록 시간 90%+ 단축
- 10만 건 규모 데드페이지 실시간 모니터링 (SemaphoreSlim + Bounded Channel 비동기 처리, 24시간 무중단)
- BigQuery → MSSQL ETL 파이프라인 (증분 적재, 품질 검증, 실패 알림 자동화)

[ML 모델링]
- XGBoost + LSTM 잔차 보정 하이브리드로 광고비 예측 MAPE 16%, ROAS 12% 개선 (SageMaker)
- 낙동강 유해남조류 예측 연구 → R²=0.880 (KCI 1저자)

기술 스택: Python, C#, SQL, JavaScript, LangChain, SageMaker, BigQuery, Power Automate, Copilot Studio, GA4, Looker Studio
자격증: SQLD, ADsP, 컴활 1급, OPIC IM3
현재 학습: 모두의연구소 AI/LLM 서비스 개발 과정 5기 (2026.02~)
</basic_info>
<basic_info_content>
""" + PROFILE_BASIC.strip() + """
</basic_info_content>

<instructions>
- 아래 참고 자료를 우선 사용하고, 기본 정보가 필요하면 위 basic_info를 참고하세요.
- 각 문단 앞의 `[참고 문서 출처 유형: …]` 표기는 인덱스에 기록된 구분입니다. 답변에서 경험이나 프로젝트를 소개할 때 **개인 프로젝트**와 **회사·기관 소속**을 구분해 말할 경우 이 표기를 따르세요. 출처가 미표시이면 둘로 단정하지 말고 사실만 전달하세요.
- 자료에 없는 내용은 추측하지 말고 "포트폴리오에 해당 내용이 없습니다"라고 하세요.
- 이전 대화 맥락이 있으면 참고하여 이어서 답하세요.
- 인사 담당자에게 소개하는 용도이므로, 답변이 자연스럽게 지원자를 어필하도록 작성하세요. (과장 없이 포트폴리오 내용을 바탕으로 강점과 경험이 잘 드러나게.)
- 답변은 한국어로, 친절하고 간결하게 작성하세요.
- 가독성을 위해 마크다운을 사용하세요: 강조는 **볼드**, 항목 나열은 - 또는 1. 2. 리스트, 소제목은 ### 등으로 구분하세요.
- 아래 <retrieval_queries>에 원문 질문과 검색에 사용한 하위 쿼리가 있습니다. 하위 쿼리 각각에 맞게 참고 자료를 활용해 답하면서, 그 내용을 모아 원문 질문에 대한 하나의 답변이 되도록 작성하세요. (하위 쿼리에 답하는 것이 곧 원문 답변을 이루게 하세요.)
</instructions>

<retrieval_queries>
- 원문 질문: {question}
- 검색에 사용한 하위 쿼리(이 관점들에 답해 원문 답변을 구성): {sub_queries}
</retrieval_queries>

<context>
{context}
</context>
"""

# RAG 불필요 포폴: 인사·메타 등 (문서 검색 없이 LLM만)
NO_RAG_PORTFOLIO_SYSTEM = """<role>당신은 송찬영을 인사·채용 담당자에게 소개하는 에이전트입니다. 지원자 송찬영은 데이터 분석가이자 LLM 에이전트 개발·설계자입니다. 인사 담당자와 대화하는 소개용 챗봇입니다.</role>
<situation>사용자(인사 담당자)의 질문은 포트폴리오 맥락이지만, 문서 검색 없이 답할 수 있습니다 (인사, 자기소개 요청 등).</situation>
<basic_info>
지원자 기본 정보: 경력(에코마케팅 인턴, 이엠넷 R&D본부(재직중)), 활동(모두의연구소 AI/LLM 과정, QMS 데이터 분석 학회, KCI 논문, 데이터청년캠퍼스 등), 학력(한국외국어대 광고PR브랜딩/통계학과 졸업).

</basic_info>
<instructions>
- 인사 담당자에게 소개하는 용도이므로, 답변이 자연스럽게 지원자를 어필하도록 작성하세요.
- 한국어로 짧고 친절하게 답하세요.
- 궁금하면 포트폴리오(경력, 강점, 프로젝트 등)에 대해 물어보라고 안내하세요.
- 가독성을 위해 **볼드**, - 리스트 등 마크다운을 적절히 사용하세요.
</instructions>"""

# 일반(포폴 무관) 질문용 (RAG 없이 LLM만)
GENERAL_SYSTEM = """<role>당신은 송찬영을 인사 담당자에게 소개하는 에이전트입니다.</role>
<situation>사용자가 포트폴리오와 무관한 질문을 했습니다.</situation>
<instructions>
- 한국어로 짧고 친절하게 답하세요.
- 궁금하면 포트폴리오(경력, 강점, 프로젝트 등)에 대해 물어보라고 안내하세요.
- 가독성을 위해 **볼드**, - 리스트 등 마크다운을 적절히 사용하세요.
</instructions>"""


def _format_sub_queries(sub_queries: list | None) -> str:
    """하위 쿼리 리스트를 프롬프트용 문자열로. 없거나 원문만 있으면 '(원문만 사용)'."""
    if not sub_queries or len(sub_queries) <= 1:
        return "(원문만 사용)"
    return "\n".join(f"{i+1}. {q.strip()}" for i, q in enumerate(sub_queries) if (q or "").strip())


def _prepare_rag_generate_input(state: dict) -> dict:
    """retrieve_pipeline 출력 state → RAG 생성 프롬프트 입력."""
    return {
        "question": state["question"],
        "chat_history": state["chat_history"],
        "context": state["context"],
        "sub_queries": _format_sub_queries(state.get("queries")),
    }


def _build_rag_gen_pipe():
    """prepare_rag_generate | generate. LangSmith: `rag_generate_pipe`."""
    prep = RunnableLambda(_prepare_rag_generate_input, name="prepare_rag_generate").with_config(
        run_name="prepare_rag_generate"
    )
    return (prep | _build_rag_generate_chain()).with_config(run_name="rag_generate_pipe")


def _eval_retry_step(state: dict) -> dict:
    """Faithfulness/Relevance 낮을 때 검색 k 증가 후 1회 재생성. `rag_quality_eval` LCEL 사용."""
    if not getattr(config, "EVAL_RETRY_ENABLED", False):
        return state
    answer = state.get("answer") or ""
    source_docs = state.get("source_docs") or []
    question = state.get("question") or ""
    chat_history = state.get("chat_history") or []
    if not source_docs:
        return state
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)
    eval_chain = build_evaluation_chain()
    eval_result = eval_chain.invoke({
        "query": question,
        "context": context.strip()[:6000] if context else "(없음)",
        "answer": answer.strip()[:3000] if answer else "(없음)",
    })
    f = eval_result.get("faithfulness_score")
    r = eval_result.get("relevance_score")
    min_f = getattr(config, "EVAL_MIN_FAITHFULNESS", 3)
    min_r = getattr(config, "EVAL_MIN_RELEVANCE", 3)
    if (f is not None and f < min_f) or (r is not None and r < min_r):
        try:
            queries = state.get("queries") or [question]
            dk = getattr(config, "HYBRID_DENSE_K", 10) + config.RETRY_K_INCREMENT
            sk = getattr(config, "HYBRID_SPARSE_K", 10) + config.RETRY_K_INCREMENT
            mn = getattr(config, "HYBRID_MERGE_TOP_N", 15)
            rag_scope = state.get("rag_scope") or "SINGLE"
            source_docs, context2, _ = _run_retrieval_core(
                question, queries, rag_scope, dense_k=dk, sparse_k=sk, merge_top_n=mn
            )
            gen = _build_rag_generate_chain()
            new_answer = gen.invoke({
                "question": question,
                "chat_history": chat_history,
                "context": context2,
                "sub_queries": _format_sub_queries(queries),
            })
            return {**state, "answer": new_answer, "source_docs": source_docs, "eval_retry_applied": True}
        except Exception:
            return state
    return state


def _rag_index_guard(state: dict) -> dict:
    """인덱스 없음·검색 실패 시 안내 문구로 통일."""
    if not state.get("source_docs") and not (state.get("context") or "").strip():
        return {
            **state,
            "answer": (
                "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요."
            ),
            "source_docs": [],
        }
    return state


def _build_entry_runnable():
    """원시 입력 → question, history_pairs, chat_history."""
    return RunnableLambda(_entry_normalize_dict, name="entry_normalize").with_config(run_name="entry_normalize")


def _entry_normalize_dict(x: dict) -> dict:
    q = (x.get("question") or "").strip()
    hp = x.get("history_pairs") or []
    return {"question": q, "history_pairs": hp, "chat_history": _pairs_to_messages(hp)}


def _router_assign_from_state(state: dict) -> dict:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return {**state, "route": "RAG"}
    try:
        out = _build_router_chain().invoke({
            "question": state["question"][:500],
            "history_text": _format_history_for_router(state["history_pairs"])[:2000],
        })
        route = out if out in ("RAG", "NO_RAG", "GENERAL") else "RAG"
        return {**state, "route": route}
    except Exception:
        return {**state, "route": "RAG"}


def _build_router_assign_runnable():
    return RunnableLambda(_router_assign_from_state, name="router_assign").with_config(run_name="router_assign")


def _wrap_text_answer(text: str) -> dict:
    return {"answer": text, "source_docs": []}


def _build_no_rag_invoke_chain(system_prompt: str, run_name: str):
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    wrap = RunnableLambda(_wrap_text_answer, name=f"{run_name}_output")
    return (prompt | llm | StrOutputParser() | wrap).with_config(run_name=run_name)


def _empty_question_response(_: dict) -> dict:
    return {"answer": "질문을 입력해 주세요.", "source_docs": []}


def _build_empty_question_runnable():
    return RunnableLambda(_empty_question_response, name="empty_question").with_config(run_name="empty_question")


def _build_rag_invoke_chain():
    """retrieve → 생성 → 품질 평가·재시도 → 인덱스 가드."""
    return (
        _build_rag_retrieve_pipeline()
        | RunnablePassthrough.assign(answer=_build_rag_gen_pipe())
        | RunnableLambda(_eval_retry_step, name="eval_retry").with_config(run_name="eval_retry")
        | RunnableLambda(_rag_index_guard, name="rag_index_guard").with_config(run_name="rag_index_guard")
    ).with_config(run_name="rag_answer_invoke")


def _build_route_branch():
    _no_rag = _build_no_rag_invoke_chain(NO_RAG_PORTFOLIO_SYSTEM, "no_rag_portfolio")
    _gen = _build_no_rag_invoke_chain(GENERAL_SYSTEM, "general_qa")
    _rag = _build_rag_invoke_chain()
    routed = (
        _build_router_assign_runnable()
        | RunnableBranch(
            (lambda x: x.get("route") == "NO_RAG", _no_rag),
            (lambda x: x.get("route") == "GENERAL", _gen),
            _rag,
        )
    )
    return RunnableBranch(
        (lambda s: not (s.get("question") or "").strip(), _build_empty_question_runnable()),
        routed,
    ).with_config(run_name="route_branch")


def _build_portfolio_answer_chain():
    """전체 포트폴리오 Q&A: entry → (빈 질문 | 라우터 → RAG/NO_RAG/GENERAL). LangSmith 루트: `portfolio_answer`."""
    return (_build_entry_runnable() | _build_route_branch()).with_config(run_name="portfolio_answer")


def _build_no_rag_stream_chain(system_prompt: str, run_name: str):
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    return (prompt | llm | StrOutputParser()).with_config(run_name=run_name)


def _build_rag_stream_chain():
    """retrieve_pipeline 출력 dict → 토큰 스트림. LangSmith: `rag_stream_chain`."""
    prep = RunnableLambda(_prepare_rag_generate_input, name="prepare_rag_generate").with_config(
        run_name="prepare_rag_generate"
    )
    return (prep | _build_rag_generate_chain()).with_config(run_name="rag_stream_chain")


def _generate_with_context(question: str, context: str, chat_history: list, sub_queries: list | None = None):
    """지정한 context로 RAG 스타일 답변 생성 (평가 후 재시도용). 원문 question과 검색에 쓴 sub_queries를 시스템 프롬프트에 포함."""
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    sub_queries_str = _format_sub_queries(sub_queries)
    return chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history,
        "sub_queries": sub_queries_str,
    })


def _invoke_no_rag(question: str, chat_history: list, system_prompt: str) -> str:
    """RAG 없이 시스템 프롬프트만으로 LLM 한 번 호출."""
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"chat_history": chat_history, "question": question})


def _stream_no_rag(question: str, chat_history: list, system_prompt: str):
    """RAG 없이 시스템 프롬프트만으로 LLM 스트리밍. (full_text_so_far) yield."""
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    stream_chain = prompt | llm | StrOutputParser()
    full = ""
    for chunk in stream_chain.stream({"chat_history": chat_history, "question": question}):
        full += chunk
        yield full


def get_chain():
    """RAG 체인 생성 (지연 로딩). 대화 히스토리 포함.
    참고: 앱의 `get_answer` / `get_answer_stream` 은 `_build_portfolio_answer_chain` 등 전체 LCEL을 사용함. 노트북·단순 검색용."""
    retriever = _load_retriever()
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["question"])),
            sub_queries=lambda x: "(원문만 사용)",  # get_chain은 Query Expansion 미사용
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def get_answer(question: str, history_pairs: list | None = None):
    """
    질문에 대해 RAG 또는 LLM만 사용. 전체 경로가 LCEL(`portfolio_answer`)로 묶여 LangSmith에서 단계별 디버깅 가능.
    Returns:
        tuple: (answer: str, source_docs: list) — 포폴 무관이면 source_docs는 []
    """
    try:
        result = _build_portfolio_answer_chain().invoke(
            {"question": question or "", "history_pairs": history_pairs or []}
        )
    except FileNotFoundError as e:
        print(f"[RAG] 인덱스 없음: {e}", flush=True)
        return (
            "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.",
            [],
        )
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        return (
            "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.",
            [],
        )
    return result.get("answer") or "", result.get("source_docs") or []


def get_answer_stream(question: str, history_pairs: list | None = None):
    """
    (full_answer_so_far, source_docs) 스트리밍. NO_RAG/GENERAL은 LCEL 스트림, RAG는 `rag_stream_pipeline` 으로 중첩 트레이스.
    """
    entry = _entry_normalize_dict({"question": question or "", "history_pairs": history_pairs or []})
    if not entry.get("question"):
        yield "질문을 입력해 주세요.", []
        return
    routed = _router_assign_from_state(entry)
    route = routed.get("route")
    _debug(f"라우터 결과: {route}")

    if route == "NO_RAG":
        chain = _build_no_rag_stream_chain(NO_RAG_PORTFOLIO_SYSTEM, "no_rag_portfolio_stream")
        full = ""
        for chunk in chain.stream({"question": routed["question"], "chat_history": routed["chat_history"]}):
            full += chunk
            yield full, []
        return
    if route == "GENERAL":
        chain = _build_no_rag_stream_chain(GENERAL_SYSTEM, "general_qa_stream")
        full = ""
        for chunk in chain.stream({"question": routed["question"], "chat_history": routed["chat_history"]}):
            full += chunk
            yield full, []
        return

    try:
        _debug("RAG 스트림: retrieve → generate")
        st = _build_rag_retrieve_pipeline().invoke(routed)
        source_docs = st.get("source_docs") or []
        ctx = (st.get("context") or "").strip()
        if not source_docs and not ctx:
            yield (
                "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.",
                [],
            )
            return
        full = ""
        for chunk in _build_rag_stream_chain().stream(st):
            full += chunk
            yield full, source_docs
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        yield (
            "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.",
            [],
        )


# ── Gradio: 전체 프로젝트 요약 기반 소개글·직무 적합성 (인덱스의 summary 청크 전체) ─────────

_PORTFOLIO_INTRO_SYSTEM = """<role>채용·인사 담당자를 위한 지원자 **소개글**을 작성하는 카피라이터입니다.</role>

<inputs>
<basic_profile> 후보 기본 프로필(보조 정보) </basic_profile>
<portfolio_summaries> 포트폴리오에서 **파일(프로젝트) 단위 요약 청크만** 모은 본문 </portfolio_summaries>
</inputs>

<rules>
- 요약·프로필에 **근거가 있는 사실만** 사용합니다. 없으면 추측하지 말고 「포트폴리오에 명시되지 않음」으로 짧게 표기합니다.
- 한국어 마크다운: ### 소제목, **강조**, 불릿.
- 톤: 격식 있고 따뜻하게, 과장 없이.
- 반드시 포함: 1) 한 줄 요약 2) 핵심 경력·학력 3) 대표 프로젝트(요약에 나온 것만) 4) 기술·역량 5) 인사 담당자에게 전하는 한마디
</rules>

<basic_profile>
{basic_profile}
</basic_profile>

<portfolio_summaries>
{summaries}
</portfolio_summaries>
"""

_JOB_FIT_SYSTEM = """<role>채용 담당자 관점에서 지원자와 **입력 직무**의 적합성을 평가합니다.</role>

<inputs>
<basic_profile> 기본 정보 </basic_profile>
<target_role> 사용자가 입력한 직무명 </target_role>
<portfolio_summaries> 프로젝트별 요약 전체 </portfolio_summaries>
</inputs>

<rules>
- 문서에 없는 사실을 지어내지 않습니다.
- 한국어 마크다운으로만 작성합니다.
- 반드시 다음 소제목을 사용합니다:
### 종합 판단
(한 문단: 적합·조건부·정보 부족 등)
### 직무와 잘 맞는 근거
(불릿, 요약·프로필 근거만)
### 부족하거나 확인이 필요한 점
### 후속 질문 제안
(면접용 2~4개)
</rules>

<basic_profile>
{basic_profile}
</basic_profile>

<target_role>
{job_title}
</target_role>

<portfolio_summaries>
{summaries}
</portfolio_summaries>
"""


def get_portfolio_summaries_context_bundle(max_chars: int | None = None) -> tuple[str, list]:
    """인덱스 bm25_docs 기준, 소스당 요약 청크 1개씩 모아 포맷된 컨텍스트 문자열과 Document 리스트."""
    max_chars = max_chars if max_chars is not None else getattr(config, "SOURCE_EXPANSION_MAX_CONTEXT_CHARS", 48_000)
    try:
        chunks = _get_corpus_chunks_list()
        if not chunks:
            return "", []
        summaries = _all_summary_docs_one_per_source(chunks)
        if not summaries:
            return "", []
        if max_chars and max_chars > 0:
            docs = _apply_char_budget_docs(summaries, max_chars)
        else:
            docs = summaries
        return _format_docs(docs), docs
    except Exception as e:
        _debug(f"get_portfolio_summaries_context_bundle: {e}")
        return "", []


def generate_intro_from_all_summaries(custom_system_prompt: str | None = None) -> str:
    """프로젝트 요약 전체를 읽고 인사 담당자용 소개글을 생성. custom_system_prompt가 있으면 기본 프롬프트 대신 사용."""
    import traceback as _tb

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "⚠️ OPENAI_API_KEY가 설정되지 않았습니다."
    ctx, _ = get_portfolio_summaries_context_bundle()
    if not ctx.strip():
        return (
            "⚠️ 인덱스에서 **프로젝트 요약 청크**를 찾을 수 없습니다. "
            "`uv run python scripts/build_index.py`로 인덱스를 다시 빌드하고, 요약 생성(INDEX_SUMMARY_ENABLED)이 켜져 있는지 확인해 주세요."
        )
    try:
        template = (custom_system_prompt or "").strip() or _PORTFOLIO_INTRO_SYSTEM
        llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.35, api_key=api_key)
        prompt = _safe_format(
            template,
            basic_profile=PROFILE_BASIC.strip()[:12_000],
            summaries=ctx[:120_000],
        )
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        _tb.print_exc()
        return f"⚠️ 소개글 생성 중 오류: {e}"


def evaluate_job_fit_for_role(job_title: str, custom_system_prompt: str | None = None) -> str:
    """전체 요약 + 프로필을 바탕으로 직무 적합성 평가 텍스트(마크다운). custom_system_prompt가 있으면 기본 프롬프트 대신 사용."""
    import traceback as _tb

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "⚠️ OPENAI_API_KEY가 설정되지 않았습니다."
    title = (job_title or "").strip()
    if not title:
        return "평가할 **직무명**을 입력해 주세요."
    ctx, _ = get_portfolio_summaries_context_bundle()
    if not ctx.strip():
        return (
            "⚠️ 인덱스에서 프로젝트 요약을 불러올 수 없습니다. `uv run python scripts/build_index.py`를 실행해 주세요."
        )
    try:
        template = (custom_system_prompt or "").strip() or _JOB_FIT_SYSTEM
        llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.2, api_key=api_key)
        prompt = _safe_format(
            template,
            basic_profile=PROFILE_BASIC.strip()[:12_000],
            job_title=title[:500],
            summaries=ctx[:120_000],
        )
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        _tb.print_exc()
        return f"⚠️ 평가 중 오류: {e}"


def get_intro_prompt_placeholder_display() -> str:
    """UI에 노출할 소개글용 시스템 프롬프트(본문 자리 표시자)."""
    return _PORTFOLIO_INTRO_SYSTEM.replace("{basic_profile}", "〈candidate_profile.py 기반 프로필〉").replace(
        "{summaries}", "〈인덱스 내 프로젝트별 요약 청크 전체·문자 상한 적용〉"
    )


def get_job_fit_prompt_placeholder_display() -> str:
    """UI에 노출할 직무 평가용 시스템 프롬프트(본문 자리 표시자)."""
    return (
        _JOB_FIT_SYSTEM.replace("{basic_profile}", "〈candidate_profile.py 기반 프로필〉")
        .replace("{job_title}", "〈사용자 입력 직무명〉")
        .replace("{summaries}", "〈인덱스 내 프로젝트별 요약 청크 전체·문자 상한 적용〉")
    )
