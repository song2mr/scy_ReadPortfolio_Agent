"""
RAG 응답 평가: 쿼리에 맞는 포트폴리오를 참고해 적절히 답했는지 LLM으로 평가.
- Faithfulness: 답변이 참고 문단(포폴)에 기반하는가 (지어내지 않았는가)
- Relevance: 질문에 맞는 답변인가, 질문과 참고 문단이 맞는가
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import sys
sys.path.insert(0, str(ROOT))
import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


EVAL_PROMPT = """<role>당신은 RAG(검색 기반 답변) 품질을 평가하는 평가자입니다.</role>

<query>{query}</query>

<reference_context>참고한 포트폴리오 문단</reference_context>
{context}

<model_answer>모델이 생성한 답변</model_answer>
{answer}

<task>
다음 두 가지를 1~5점으로 평가하고, 각각 한 줄 이유를 작성하세요.
</task>
<criteria>
- Faithfulness(충실도): 답변이 참고 문단에만 기반하는가? 문단에 없는 내용을 지어내지 않았는가? (5=완전히 기반, 1=많이 지어냄)
- Relevance(적합성): 질문에 맞는 답변인가? 참고 문단이 질문과 관련 있는가? (5=매우 적합, 1=무관함)
</criteria>

<output_format>반드시 아래 형식만 사용하세요.</output_format>
FAITHFULNESS: [1-5]
REASON_F: [한 줄 이유]
RELEVANCE: [1-5]
REASON_R: [한 줄 이유]
"""


def _parse_eval_output(text: str) -> dict:
    """평가 LLM 출력에서 점수·이유 추출."""
    out = {
        "faithfulness_score": None,
        "faithfulness_reason": "",
        "relevance_score": None,
        "relevance_reason": "",
    }
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("FAITHFULNESS:"):
            m = re.search(r"\b([1-5])\b", line)
            if m:
                out["faithfulness_score"] = int(m.group(1))
        elif line.upper().startswith("REASON_F:"):
            out["faithfulness_reason"] = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("RELEVANCE:"):
            m = re.search(r"\b([1-5])\b", line)
            if m:
                out["relevance_score"] = int(m.group(1))
        elif line.upper().startswith("REASON_R:"):
            out["relevance_reason"] = line.split(":", 1)[-1].strip()
    return out


def build_evaluation_chain(model: str | None = None):
    """Faithfulness / Relevance 평가용 LCEL. LangSmith에서 `rag_quality_eval` 단계로 표시."""
    prompt = ChatPromptTemplate.from_template(EVAL_PROMPT)
    llm = ChatOpenAI(
        model=model or config.OPENAI_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def _parse_msg(msg) -> dict:
        text = msg.content if hasattr(msg, "content") else str(msg)
        return _parse_eval_output(text)

    return (prompt | llm | RunnableLambda(_parse_msg)).with_config(run_name="rag_quality_eval")


def evaluate_response(
    query: str,
    context: str,
    answer: str,
    model: str | None = None,
) -> dict:
    """
    한 번의 (질문, 참고 문단, 답변)에 대해 Faithfulness / Relevance 를 LLM으로 평가.

    Args:
        query: 사용자 질문
        context: RAG가 참고한 포트폴리오 문단 (합친 문자열)
        answer: RAG가 생성한 답변
        model: 평가용 LLM 모델 (기본 config.OPENAI_MODEL)

    Returns:
        dict: faithfulness_score(1-5), faithfulness_reason, relevance_score(1-5), relevance_reason
    """
    model = model or config.OPENAI_MODEL
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "faithfulness_score": None,
            "faithfulness_reason": "OPENAI_API_KEY 없음",
            "relevance_score": None,
            "relevance_reason": "OPENAI_API_KEY 없음",
        }
    # 문단/답변이 너무 길면 자르기
    context_trim = context.strip()[:6000] if context else "(없음)"
    answer_trim = answer.strip()[:3000] if answer else "(없음)"
    try:
        chain = build_evaluation_chain(model=model)
        return chain.invoke({
            "query": query,
            "context": context_trim,
            "answer": answer_trim,
        })
    except Exception as e:
        return {
            "faithfulness_score": None,
            "faithfulness_reason": f"평가 오류: {e}",
            "relevance_score": None,
            "relevance_reason": f"평가 오류: {e}",
        }


def evaluate_response_from_docs(
    query: str,
    source_docs: list,
    answer: str,
    model: str | None = None,
) -> dict:
    """
    Document 리스트와 답변을 받아 context 문자열로 합친 뒤 evaluate_response 호출.
    """
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)
    return evaluate_response(query=query, context=context, answer=answer, model=model)
