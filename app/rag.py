"""
RAG 체인: 포폴 관련 여부 라우팅, FAISS 검색 → OpenAI 답변 생성. 대화 히스토리 반영.
"""
import os
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder

from app.rag_eval import evaluate_response_from_docs


def _format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


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


def _ensure_index_dir():
    """로컬에 인덱스가 없고 HF_INDEX_DATASET 이 설정되어 있으면 Dataset에서 index/ 로 다운로드."""
    index_dir = Path(config.INDEX_DIR)
    if (index_dir / "index.faiss").exists():
        return
    hf_dataset = os.getenv("HF_INDEX_DATASET", "").strip()
    if not hf_dataset:
        return
    try:
        from huggingface_hub import snapshot_download
        _debug(f"HF Dataset에서 인덱스 다운로드 중: {hf_dataset}")
        index_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_dataset, repo_type="dataset", local_dir=str(index_dir))
        _debug("인덱스 다운로드 완료")
    except Exception as e:
        _debug(f"HF 인덱스 다운로드 실패: {e}")


def _load_retriever(k: int | None = None):
    """k 미지정 시 Reranker 활성이면 RETRIEVE_K_INITIAL, 아니면 RETRIEVE_K 사용."""
    if k is not None:
        use_k = k
    elif getattr(config, "RERANKER_ENABLED", False):
        use_k = getattr(config, "RETRIEVE_K_INITIAL", 15)
    else:
        use_k = config.RETRIEVE_K
    _ensure_index_dir()
    _debug("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    _debug("FAISS 인덱스 로드 중...")
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _debug("retriever 준비 완료")
    return vectorstore.as_retriever(search_kwargs={"k": use_k})


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


def _rerank_docs(query: str, docs: list, top_n: int | None = None) -> list:
    """Cross-Encoder로 문서를 재순위화하여 상위 top_n개만 반환.
    Reranker 비활성이거나 문서가 적으면 원본 그대로 반환."""
    reranker = _load_reranker()
    if reranker is None or not docs:
        return docs
    top_n = top_n or getattr(config, "RERANKER_TOP_N", 4)
    if len(docs) <= top_n:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored_docs[:top_n]]
    _debug(f"Rerank 완료: {len(docs)}개 → {len(reranked)}개 (scores: {sorted(scores, reverse=True)[:5]})")
    return reranked


# 라우터: RAG 필요 포폴 / RAG 불필요 포폴 / 일반 — 한 번에 구분 (대화 히스토리 참고)
ROUTER_PROMPT = """<task>
다음 질문을 아래 세 가지 중 하나로만 분류하세요. 대화 맥락이 있으면 이를 참고하세요.

- RAG: 지원자 포트폴리오(이력서·경력·프로젝트·학력 등) 문서를 검색해서 구체적 내용으로 답해야 하는 질문. (예: 주요 프로젝트?, 강점?, 어디 근무? / 이전에 경력 이야기 중이면 "프로젝트는?"도 RAG)
- NO_RAG: 포트폴리오 Q&A 맥락이지만 문서 검색 없이 답할 수 있는 질문. (예: 안녕, 너 누구야?, 뭐 할 수 있어?)
- GENERAL: 포트폴리오와 무관한 질문. (예: 날씨, 일반 상식, 인사 이야기)
</task>
<output>반드시 RAG, NO_RAG, GENERAL 중 한 단어만 출력하세요.</output>

<recent_conversation>
{history_text}
</recent_conversation>

<current_question>{question}</current_question>"""


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


def _route_question(question: str, history_pairs: list | None = None) -> str:
    """질문을 RAG / NO_RAG / GENERAL 중 하나로 분류 (대화 히스토리 참고, LLM 한 번 호출)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "RAG"
    history_pairs = history_pairs or []
    history_text = _format_history_for_router(history_pairs)
    try:
        router_model = getattr(config, "OPENAI_MODEL_ROUTER", config.OPENAI_MODEL)
        llm = ChatOpenAI(model=router_model, temperature=0, api_key=api_key)
        out = llm.invoke(
            _safe_format(
                ROUTER_PROMPT,
                question=question[:500],
                history_text=history_text[:2000],
            )
        ).content
        raw = out.strip().upper()
        if "NO_RAG" in raw or "NO RAG" in raw:
            return "NO_RAG"
        if "GENERAL" in raw:
            return "GENERAL"
        return "RAG"
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
- 답변은 한국어로, 친절하고 간결하게 작성하세요.
- 가독성을 위해 마크다운을 사용하세요: 강조는 **볼드**, 항목 나열은 - 또는 1. 2. 리스트, 소제목은 ### 등으로 구분하세요.
</instructions>

<context>
{context}
</context>
"""

# RAG 불필요 포폴: 인사·메타 등 (문서 검색 없이 LLM만)
NO_RAG_PORTFOLIO_SYSTEM = """<role>당신은 송찬영을 인사·채용 담당자에게 소개하는 에이전트입니다. 지원자 송찬영은 데이터 분석가이자 LLM 에이전트 개발·설계자입니다. 인사 담당자와 대화하는 소개용 챗봇입니다.</role>
<situation>사용자(인사 담당자)의 질문은 포트폴리오 맥락이지만, 문서 검색 없이 답할 수 있습니다 (인사, 자기소개 요청 등).</situation>
<basic_info>지원자 기본 정보: 경력(에코마케팅 인턴, 이엠넷 R&D본부), 활동(모두의연구소 AI/LLM 과정, QMS 데이터 분석 학회, KCI 논문, 데이터청년캠퍼스 등), 학력(한국외국어대 광고PR브랜딩/통계학과 졸업).</basic_info>
<instructions>
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


def _generate_with_context(question: str, context: str, chat_history: list):
    """지정한 context로 RAG 스타일 답변 생성 (평가 후 재시도용)."""
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
    return chain.invoke({"question": question, "context": context, "chat_history": chat_history})


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
    """RAG 체인 생성 (지연 로딩). 대화 히스토리 포함."""
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
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# 싱글톤처럼 한 번만 로드 (앱 기동 시)
_chain = None
_retriever = None


def get_answer(question: str, history_pairs: list | None = None):
    """
    질문에 대해 RAG 또는 LLM만 사용. 대화 히스토리 반영.
    Returns:
        tuple: (answer: str, source_docs: list) — 포폴 무관이면 source_docs는 []
    """
    global _chain, _retriever
    history_pairs = history_pairs or []
    chat_history = _pairs_to_messages(history_pairs)

    route = _route_question(question, history_pairs)
    if route == "NO_RAG":
        answer = _invoke_no_rag(question, chat_history, NO_RAG_PORTFOLIO_SYSTEM)
        return answer, []
    if route == "GENERAL":
        answer = _invoke_no_rag(question, chat_history, GENERAL_SYSTEM)
        return answer, []

    # RAG → 검색 → Reranker → context로 답변
    try:
        if _retriever is None:
            _, _retriever = get_chain()
    except Exception:
        return "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []
    source_docs = _retriever.invoke(question)
    source_docs = _rerank_docs(question, source_docs)
    context = _format_docs(source_docs)
    answer = _generate_with_context(question, context, chat_history)

    # 평가 후 재시도: Faithfulness/Relevance 낮으면 k 늘려 1회만 재검색·재생성
    if (
        getattr(config, "EVAL_RETRY_ENABLED", False)
        and source_docs
    ):
        eval_result = evaluate_response_from_docs(question, source_docs, answer)
        f = eval_result.get("faithfulness_score")
        r = eval_result.get("relevance_score")
        min_f = getattr(config, "EVAL_MIN_FAITHFULNESS", 3)
        min_r = getattr(config, "EVAL_MIN_RELEVANCE", 3)
        if (f is not None and f < min_f) or (r is not None and r < min_r):
            try:
                retry_k = getattr(config, "RETRIEVE_K_RETRY", 8)
                retriever_retry = _load_retriever(k=retry_k)
                source_docs = retriever_retry.invoke(question)
                source_docs = _rerank_docs(question, source_docs)
                context = _format_docs(source_docs)
                answer = _generate_with_context(question, context, chat_history)
            except Exception:
                pass  # 재시도 실패 시 첫 답변 유지

    return answer, source_docs


def get_answer_stream(question: str, history_pairs: list | None = None):
    """
    질문에 대해 RAG 또는 LLM만 스트리밍. (full_answer_so_far, source_docs) 를 yield.
    포폴 무관이면 source_docs는 [].
    """
    global _chain, _retriever
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

    # RAG → 스트리밍 (인덱스 없으면 에러 메시지 한 번 yield)
    try:
        if _retriever is None:
            _debug("첫 RAG 요청: 인덱스 로드 중 (최초 1회, 수십 초 걸릴 수 있음)...")
            _, _retriever = get_chain()
            _debug("인덱스 로드 완료")
        _debug("검색 중...")
        source_docs = _retriever.invoke(question)
        _debug(f"검색 완료 (문단 {len(source_docs)}개)")
        source_docs = _rerank_docs(question, source_docs)
        _debug(f"Rerank 후 문단 {len(source_docs)}개")
    except Exception:
        yield "포트폴리오 인덱스가 없습니다. 터미널에서 `uv run python scripts/build_index.py` 를 먼저 실행해 주세요.", []
        return
    context = _format_docs(source_docs)
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
    stream_chain = prompt | llm | StrOutputParser()
    full = ""
    _debug("LLM 스트리밍 시작...")
    for chunk in stream_chain.stream({
        "question": question,
        "context": context,
        "chat_history": chat_history,
    }):
        full += chunk
        yield full, source_docs
