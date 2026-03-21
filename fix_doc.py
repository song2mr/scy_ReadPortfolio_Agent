# -*- coding: utf-8 -*-
p = "docs/코드_동작_설명.md"
with open(p, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Replace 5.2 steps 2-6: find "2. **포폴 관련이면" or "2. **RAG 분기" then next lines until "6. **반환**"
out = []
i = 0
while i < len(lines):
    line = lines[i]
    if i + 1 < len(lines) and "2. **RAG 분기: _expand_query(question)**" in line:
        # Already updated step 2 title; replace the bullet and steps 3-8
        out.append(line)
        if "질문만 LLM에 넣어 재작성" in lines[i + 1]:
            i += 1
            out.append(lines[i])
            i += 1
            # Skip old 3-6, inject new 3-8
            while i < len(lines) and not ("8. **반환**" in lines[i] and "출처 문단)`" in lines[i + 1]):
                if "### 5.3" in lines[i]:
                    break
                i += 1
            new_steps = """3. **쿼리별 _hybrid_retrieve**  
   - 각 쿼리마다 Dense(FAISS) top-k + Sparse(BM25/Kiwi) top-k 검색 → RRF로 병합해 쿼리당 N개 랭킹.
4. **_rrf_merge_multiple(rankings)**  
   - 여러 쿼리의 랭킹을 다시 RRF로 병합해 상위 N개 선정.
5. **_rerank_docs(question, merged)**  
   - Cross-Encoder로 재순위화해 상위 4개만 선별.
6. **context + 프롬프트 + LLM**  
   - 선별된 문단을 context로 붙이고, 아래 참고 자료만 사용해서 답하라 프롬프트로 OpenAI 호출.
7. **평가 후 재시도 (옵션)**  
   - EVAL_RETRY_ENABLED 이고 source_docs 가 있으면 Faithfulness/Relevance 평가. 최소 점수 미만이면 k 늘려 1회 재검색·재생성.
8. **반환**  
   - `(답변 문자열, 출처 문단)` 를 앱에 넘긴다.
"""
            out.append(new_steps)
            if i < len(lines) and "6. **반환**" in lines[i]:
                i += 2  # skip "6. **반환**" and its bullet
            continue
    if "2. **포폴 관련이면 retriever.invoke(question)**" in line or (
        "2. **RAG 분기" in line and "질문 문장을 임베딩" in (lines[i + 1] if i + 1 < len(lines) else "")
    ):
        out.append("2. **RAG 분기: _expand_query(question)**  \n")
        out.append("   - 질문만 LLM에 넣어 재작성. 단일 → 1줄, 포괄적 → 2~4개 하위 질문. [원본, 재작성1, ...] 리스트 생성.\n")
        i += 2
        while i < len(lines):
            if "### 5.3" in lines[i]:
                break
            if "6. **반환**" in lines[i] and i + 1 < len(lines) and "출처 문단 4개" in lines[i + 1]:
                out.append("3. **쿼리별 _hybrid_retrieve**  \n")
                out.append("   - 각 쿼리마다 Dense(FAISS) top-k + Sparse(BM25/Kiwi) top-k 검색 → RRF로 병합해 쿼리당 N개 랭킹.\n")
                out.append("4. **_rrf_merge_multiple(rankings)**  \n")
                out.append("   - 여러 쿼리의 랭킹을 다시 RRF로 병합해 상위 N개 선정.\n")
                out.append("5. **_rerank_docs(question, merged)**  \n")
                out.append("   - Cross-Encoder로 재순위화해 상위 4개만 선별.\n")
                out.append("6. **context + 프롬프트 + LLM**  \n")
                out.append("   - 선별된 문단을 context로 붙이고, 아래 참고 자료만 사용해서 답하라 프롬프트로 OpenAI 호출.\n")
                out.append("7. **평가 후 재시도 (옵션)**  \n")
                out.append("   - EVAL_RETRY_ENABLED 이고 source_docs 가 있으면 Faithfulness/Relevance 평가. 최소 점수 미만이면 k 늘려 1회 재검색·재생성.\n")
                out.append("8. **반환**  \n")
                out.append("   - `(답변 문자열, 출처 문단)` 를 앱에 넘긴다.\n")
                i += 2
                break
            i += 1
        continue
    out.append(line)
    i += 1

with open(p, "w", encoding="utf-8") as f:
    f.writelines(out)
print("5.2 steps updated")
