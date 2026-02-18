"""
RAG 평가 스크립트: 샘플 쿼리로 답변 생성 후 Faithfulness/Relevance 평가.
실행: 프로젝트 루트에서 uv run python scripts/evaluate_rag.py
인덱스가 있어야 함 (uv run python scripts/build_index.py 선행).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import config
from app.rag import get_answer
from app.rag_eval import evaluate_response_from_docs


# 평가할 샘플 쿼리 (원하면 파일에서 읽거나 추가)
SAMPLE_QUERIES = [
    "이 사람의 강점을 한 줄로 요약해 주세요.",
    "주요 경력과 프로젝트를 간단히 소개해 주세요.",
    "어떤 역할/포지션에 적합해 보이나요?",
]


def main():
    if not (config.INDEX_DIR / "index.faiss").exists():
        print("인덱스가 없습니다. 먼저 uv run python scripts/build_index.py 를 실행하세요.")
        return

    print("RAG 평가 (샘플 쿼리 → 답변 → Faithfulness/Relevance)\n")
    results = []

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"[{i}/{len(SAMPLE_QUERIES)}] 질문: {query[:50]}...")
        try:
            answer, source_docs = get_answer(query, history_pairs=[])
            if not source_docs:
                print("  → 포폴 무관으로 분류되어 RAG 미사용. 평가 생략.\n")
                results.append({
                    "query": query,
                    "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
                    "eval": "RAG 미사용(라우팅)",
                })
                continue
            eval_result = evaluate_response_from_docs(query, source_docs, answer)
            results.append({
                "query": query,
                "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
                "eval": eval_result,
            })
            print(f"  Faithfulness: {eval_result.get('faithfulness_score')}/5 - {eval_result.get('faithfulness_reason', '')}")
            print(f"  Relevance:    {eval_result.get('relevance_score')}/5 - {eval_result.get('relevance_reason', '')}")
        except Exception as e:
            print(f"  오류: {e}")
            results.append({"query": query, "error": str(e)})
        print()

    print("--- 요약 ---")
    evals = [r.get("eval") for r in results if isinstance(r.get("eval"), dict)]
    if evals:
        f_scores = [e.get("faithfulness_score") for e in evals if e.get("faithfulness_score") is not None]
        r_scores = [e.get("relevance_score") for e in evals if e.get("relevance_score") is not None]
        if f_scores:
            print(f"Faithfulness 평균: {sum(f_scores) / len(f_scores):.2f}/5")
        if r_scores:
            print(f"Relevance 평균:    {sum(r_scores) / len(r_scores):.2f}/5")


if __name__ == "__main__":
    main()
