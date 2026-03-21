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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from app.rag_eval import evaluate_response_from_docs

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


def _format_docs(docs):
    return "\n\n---\n\n".join((doc.page_content or "") for doc in docs)


def _strip_html(text) -> str:
    """Gradio 등에서 bot 이 리스트로 올 수 있음 → 문자열로 통일 후 처리."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _pairs_to_messages(pairs: list, max_turns: int = 5) -> list:
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


# Query Expansion: 질문만으로 LLM 재작성 (1차 검색 없음. 단일/포괄적 질문 → 1개 또는 여러 하위 쿼리)
# 포트폴리오·경험 조회 시 쿼리 재작성에 참고할 프로젝트·솔루션 목록 (구체적 키워드로 활용)
QUERY_EXPANSION_PORTFOLIO_TOPICS = """
## 생성형 AI 프로젝트
- 머신러닝을 활용한 낙동강 유해남조류 발생 예측
- 로그 트래킹 스크립트, SEO & AEO 컨설팅 및 설계
- MS Copilot Studio AI 사용 정책 수립

## 자동화 솔루션 및 머신러닝 모델 개발
- 애드메이커: 광고 자동 생성 시스템 구축
- 광고 데드페이지 실시간 모니터링 시스템 개발
- 예산 소진 예측 모델 설계(SageMaker)
- 마케팅 데이터 ETL 파이프라인 구축
- 포트폴리오 에이전트(개인 프로젝트)
- 광고 데이터 분석 챗봇 설계
- 사내 솔루션 Q&A 챗봇 설계
- ChatGPT를 이용한 광고 소재 제작 솔루션
"""

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
            portfolio_topics=QUERY_EXPANSION_PORTFOLIO_TOPICS.strip(),
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


def _format_history_for_router(pairs: list, max_turns: int = 5, max_chars_per_turn: int = 200) -> str:
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
        portfolio_topics=QUERY_EXPANSION_PORTFOLIO_TOPICS.strip()
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


def _retrieve_step(state: dict) -> dict:
    """LCEL 내부: question + queries → hybrid 검색, RRF, rerank 후 context/source_docs 추가. LangSmith에 검색 결과 노출."""
    question = state.get("question") or ""
    queries = state.get("queries") or [question]
    try:
        vs = _get_vectorstore()
        bm25 = _get_bm25()
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        return {**state, "context": "", "source_docs": [], "retrieved_preview": []}
    dense_k = getattr(config, "HYBRID_DENSE_K", 10)
    sparse_k = getattr(config, "HYBRID_SPARSE_K", 10)
    merge_top_n = getattr(config, "HYBRID_MERGE_TOP_N", 15)
    rankings = [_hybrid_retrieve(q, vs, bm25, dense_k, sparse_k, merge_top_n) for q in queries]
    merged = _rrf_merge_multiple(rankings, top_n=merge_top_n)
    source_docs = _rerank_docs(question, merged)
    context = _format_docs(source_docs)
    # LangSmith에서 "무슨 문서를 긁어왔는지" 보이도록 요약 추가
    retrieved_preview = [(doc.page_content or "")[:300] for doc in source_docs]
    return {
        **state,
        "context": context,
        "source_docs": source_docs,
        "retrieved_preview": retrieved_preview,
    }


def _build_retrieve_chain():
    """LCEL: RunnableLambda(retrieve_step). 입력 {question, queries} → 출력에 context, source_docs, retrieved_preview 포함."""
    return RunnableLambda(_retrieve_step, name="retrieve").with_config(run_name="retrieve")


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


@traceable(name="router", run_type="chain")
def _route_question(question: str, history_pairs: list | None = None) -> str:
    """질문을 RAG / NO_RAG / GENERAL 중 하나로 분류. LCEL router_chain 사용 (LangSmith 트레이스)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "RAG"
    history_pairs = history_pairs or []
    history_text = _format_history_for_router(history_pairs)
    try:
        router_chain = _build_router_chain()
        out = router_chain.invoke({
            "question": question[:500],
            "history_text": history_text[:2000],
        })
        return out if out in ("RAG", "NO_RAG", "GENERAL") else "RAG"
    except Exception:
        return "RAG"


# 지원자 기본 정보 (시스템 프롬프트에 포함 — RAG 검색 보조)
PROFILE_BASIC = """
#경력 사항
2023.01~04 에코마케팅 인턴
2024.01~현재 이엠넷, R&D본부: 진행 중
#활동 내역
2026.02~현재 모두의연구소 AI/LLM 서비스 개발 과정: 진행 중
2023.06~12 QMS 데이터 분석 학회: 끝
2023.02 KCI 논문 등재(1저자)
2022.06~08 데이터청년캠퍼스: 끝
2021.01~03 네이버게임 서포터즈: 끝
#소프트웨어
Python R Java Script GA SQL Google Big Query C# Looker Studio
#학력 사항
2018.03 한국외국어 대학교 서울캠퍼스 ~ 2024.06 광고PR브랜딩/통계학과 졸업 (3.92/4.5)
#강점 (짤막 요약)
- 로그 트래킹·SEO/AEO 수집 설계, ETL 파이프라인, ML·RAG 활용까지 데이터 전체 라이프사이클을 끊김 없이 연결해 와서, 병목 지점을 구조적으로 판단할 수 있음.
- LLM을 챗봇·생성형 콘텐츠·자동화 등 실무에 적용한 설계·운영 경험이 있으며, 어디에 AI를 쓰면 의미 있고 어디선 복잡해지는지 시행착오로 학습함.
"""

# 시스템 프롬프트: 송찬영을 인사 담당자에게 소개하는 에이전트 (송찬영 = 데이터 분석가·LLM 에이전트 개발·설계자)
RAG_SYSTEM = """<role>당신은 송찬영을 인사·채용 담당자에게 소개하는 에이전트입니다. 지원자 송찬영은 데이터 분석가이자 LLM 에이전트 개발·설계자입니다. 당신은 그 포지션의 어시스턴트가 아니라, 인사 담당자가 질문하면 송찬영의 포트폴리오(이력서·경력기술서·프로젝트)를 RAG로 검색해 답변하는 소개용 에이전트입니다.</role>

<basic_info>아래는 지원자(송찬영) 기본 정보입니다. 참고 자료와 함께 활용하세요.</basic_info>
<basic_info_content>
""" + PROFILE_BASIC.strip() + """
</basic_info_content>

<instructions>
- 아래 참고 자료를 우선 사용하고, 기본 정보가 필요하면 위 basic_info를 참고하세요.
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
<basic_info>지원자 기본 정보: 경력(에코마케팅 인턴, 이엠넷 R&D본부), 활동(모두의연구소 AI/LLM 과정, QMS 데이터 분석 학회, KCI 논문, 데이터청년캠퍼스 등), 학력(한국외국어대 광고PR브랜딩/통계학과 졸업).</basic_info>
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


@traceable(name="generate_with_context", run_type="chain")
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
    참고: get_answer/get_answer_stream은 이 체인을 쓰지 않고 Query Expansion + Hybrid Search 경로를 사용함. 노트북·외부 호출용."""
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


def _build_full_rag_chain():
    """LCEL: query_expansion | retrieve | generate. LangSmith에 각 단계·검색된 문서(retrieved_preview) 노출."""
    return (
        _build_query_expansion_chain()
        | _build_retrieve_chain()
        | RunnablePassthrough.assign(
            answer=lambda s: _build_rag_generate_chain().invoke({
                "question": s["question"],
                "chat_history": s["chat_history"],
                "context": s["context"],
                "sub_queries": _format_sub_queries(s.get("queries")),
            }),
        )
    ).with_config(run_name="rag")


@traceable(name="rag")
def get_answer(question: str, history_pairs: list | None = None):
    """
    질문에 대해 RAG 또는 LLM만 사용. 대화 히스토리 반영.
    LCEL 구조(router | query_expansion | retrieve | generate)로 LangSmith에 단계·검색 문서 노출.
    Returns:
        tuple: (answer: str, source_docs: list) — 포폴 무관이면 source_docs는 []
    """
    if not (question or "").strip():
        return "질문을 입력해 주세요.", []
    question = (question or "").strip()
    history_pairs = history_pairs or []
    chat_history = _pairs_to_messages(history_pairs)

    route = _route_question(question, history_pairs)
    if route == "NO_RAG":
        answer = _invoke_no_rag(question, chat_history, NO_RAG_PORTFOLIO_SYSTEM)
        return answer, []
    if route == "GENERAL":
        answer = _invoke_no_rag(question, chat_history, GENERAL_SYSTEM)
        return answer, []

    # RAG: LCEL query_expansion | retrieve | generate (LangSmith에 검색 내용·단계 노출)
    try:
        full_rag = _build_full_rag_chain()
        state0 = {"question": question, "chat_history": chat_history}
        result = full_rag.invoke(state0)
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        return "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []

    source_docs = result.get("source_docs") or []
    answer = result.get("answer") or ""
    if not source_docs and not (result.get("context") or "").strip():
        return "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []

    # 평가 후 재시도: Faithfulness/Relevance 낮으면 k 늘려 1회만 재검색·재생성
    if getattr(config, "EVAL_RETRY_ENABLED", False) and source_docs:
        eval_result = evaluate_response_from_docs(question, source_docs, answer)
        f = eval_result.get("faithfulness_score")
        r = eval_result.get("relevance_score")
        min_f = getattr(config, "EVAL_MIN_FAITHFULNESS", 3)
        min_r = getattr(config, "EVAL_MIN_RELEVANCE", 3)
        if (f is not None and f < min_f) or (r is not None and r < min_r):
            try:
                vs = _get_vectorstore()
                bm25 = _get_bm25()
                queries = result.get("queries") or [question]
                dense_k = getattr(config, "HYBRID_DENSE_K", 10) + 5
                sparse_k = getattr(config, "HYBRID_SPARSE_K", 10) + 5
                merge_top_n = getattr(config, "HYBRID_MERGE_TOP_N", 15)
                retry_rankings = [_hybrid_retrieve(q, vs, bm25, dense_k, sparse_k, merge_top_n) for q in queries]
                merged_retry = _rrf_merge_multiple(retry_rankings, top_n=merge_top_n)
                source_docs = _rerank_docs(question, merged_retry)
                context = _format_docs(source_docs)
                answer = _generate_with_context(question, context, chat_history, sub_queries=queries)
            except Exception:
                pass

    return answer, source_docs


@traceable(name="rag_stream")
def get_answer_stream(question: str, history_pairs: list | None = None):
    """
    질문에 대해 RAG 또는 LLM만 스트리밍. (full_answer_so_far, source_docs) 를 yield.
    LCEL query_expansion | retrieve 사용 후 generate만 스트리밍 (LangSmith 트레이스).
    """
    if not (question or "").strip():
        yield "질문을 입력해 주세요.", []
        return
    question = (question or "").strip()
    history_pairs = history_pairs or []
    chat_history = _pairs_to_messages(history_pairs)

    _debug("라우터 분류 중...")
    route = _route_question(question, history_pairs)
    _debug(f"라우터 결과: {route}")
    if route == "NO_RAG":
        for full in _stream_no_rag(question, chat_history, NO_RAG_PORTFOLIO_SYSTEM):
            yield full, []
        return
    if route == "GENERAL":
        for full in _stream_no_rag(question, chat_history, GENERAL_SYSTEM):
            yield full, []
        return

    # RAG: LCEL query_expansion | retrieve (트레이스) → generate 스트리밍
    try:
        _debug("Query Expansion·검색 체인 실행...")
        expansion_chain = _build_query_expansion_chain()
        retrieve_chain = _build_retrieve_chain()
        state0 = {"question": question, "chat_history": chat_history}
        state1 = expansion_chain.invoke(state0)
        state2 = retrieve_chain.invoke(state1)
        source_docs = state2.get("source_docs") or []
        context = state2.get("context") or ""
        queries = state2.get("queries") or [question]
        if not source_docs and not context.strip():
            yield "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []
            return
        _debug("LLM 스트리밍 시작...")
        stream_chain = _build_rag_generate_chain()
        full = ""
        for chunk in stream_chain.stream({
            "question": question,
            "chat_history": chat_history,
            "context": context,
            "sub_queries": _format_sub_queries(queries),
        }):
            full += chunk
            yield full, source_docs
    except Exception as e:
        print(f"[RAG] 인덱스 로드 실패: {e}", flush=True)
        yield "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []
        return
