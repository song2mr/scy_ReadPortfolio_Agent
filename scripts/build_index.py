"""
포트폴리오 문서를 로드·청킹·임베딩 후 FAISS + BM25(Kiwi) 인덱스를 index/ 에 저장합니다.
HF 업로드 시 index/ 폴더 전체(index.faiss, index.pkl, bm25_*.pkl)를 올리면 됩니다.
- 로컬: data/portfolio/ 의 PDF, DOCX, Markdown(.md) 사용
- 구글 드라이브: GOOGLE_DRIVE_FOLDER_ID + INDEX_BUILD_USE_LOCAL_ONLY=false 일 때만 드라이브에서 로드 (기본은 로컬만)
실행: 프로젝트 루트에서 uv run python scripts/build_index.py
"""
import os
import pickle
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

import config
from app.portfolio_origins import (
    load_portfolio_origins_config,
    metadata_for_origin,
    resolve_portfolio_origin_drive,
    resolve_portfolio_origin_local,
)


def _doc_source_key(doc: Document) -> str:
    md = doc.metadata or {}
    return (md.get("source") or md.get("title") or "").strip() or "__missing__"


def _merge_doc_metadata(docs: list, extra: dict) -> None:
    if not extra:
        return
    for d in docs:
        md = dict(d.metadata or {})
        md.update(extra)
        d.metadata = md


def _group_docs_by_source_ordered(docs: list) -> tuple[list[str], dict[str, list]]:
    """로드 순서 유지: 각 source가 처음 나온 순서대로 키 목록 반환."""
    order: list[str] = []
    groups: dict[str, list] = {}
    for d in docs:
        k = _doc_source_key(d)
        if k not in groups:
            order.append(k)
            groups[k] = []
        groups[k].append(d)
    return order, groups


def _concat_source_text(doc_list: list) -> str:
    parts = []
    for d in doc_list:
        t = (d.page_content or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


SUMMARY_PROMPT = """당신은 포트폴리오 문서에서 **RAG 검색용 요약 청크**를 만드는 편집자입니다. 아래 텍스트는 **한 개 파일(한 프로젝트/문서 단위)** 의 내용입니다.
검색 시 사용자가 쓸 법한 **구체 명사·프로젝트명·제품명·기술명·도메인 용어**가 요약 안에 자연스럽게 들어가야 합니다. (BM25·벡터 검색이 질문과 이어지도록)

## 반드시 반영할 정보(문서에 있을 때만, 빠뜨리지 말 것)
다음 항목이 본문에 **명시되어 있으면** 요약 본문에 꼭 넣으세요. 없으면 쓰지 마세요.
1) **무엇을** 만들거나 다루었는지(서비스·과제·역할 한 줄)
2) **기술 스택·도구·프레임워크·모델·DB·인프라** (약어뿐 아니라 풀네임이 있으면 둘 다 쓰기 좋음)
3) **데이터·규모·환경**이 드러나는 수치나 조건(있다면)
4) **성과·지표·검증·배포** (숫자·범위가 있으면 그대로)
5) **범위 한계·특이사항**(문서가 강조한 제약·역할 분담 등)

## 작성 규칙
- **4~7문장**. 첫 문장에 프로젝트/주제를 한 번에 짚을 수 있게 쓰세요.
- 문장마다 **검색에 걸릴 만한 실체어**(고유명사, 기술명, 업무 도메인)를 골고루 넣으세요. 추상적인 말만 나열하지 마세요.
- 본문에 **없는 추측·과장·일반론**은 넣지 마세요. 불확실하면 쓰지 마세요.
- **목록·표**에만 있던 핵심 항목이 있으면 요약 문장 속에 녹여 넣으세요.

## 출력 형식(반드시 지킬 것)
1) 위 규칙에 맞는 **연속 한 덩어리 한국어 요약** (문장들)
2) 마지막에 **빈 줄 하나** 다음, 아래 두 줄을 **정확히** 이 형식으로:
키워드: 검색용 쉼표구분 나열 (기술·도메인·역할·제품명 등, 8~20개 정도)
동의어: 사용자가 쓸 법한 다른 표현·약어를 쉼표로 (본문·상식 범위 내에서만, 없으면 "동의어: (해당 없음)")

[파일]
{source_label}

[본문]
{document_text}
"""


def _build_summary_document(source_key: str, doc_list: list) -> Document | None:
    """파일 단위 LLM 요약 → chunk_kind=summary. 실패 시 None."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    text = _concat_source_text(doc_list)
    if not text.strip():
        return None
    max_in = getattr(config, "INDEX_SUMMARY_MAX_INPUT_CHARS", 14_000)
    if len(text) > max_in:
        text = text[:max_in] + "\n\n...(이후 생략)"
    label = source_key
    try:
        label_path = Path(source_key)
        if label_path.suffix:
            label = label_path.name
    except Exception:
        pass
    prompt = SUMMARY_PROMPT.format(source_label=label, document_text=text)
    try:
        llm = ChatOpenAI(
            model=getattr(config, "INDEX_SUMMARY_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )
        out = llm.invoke(prompt)
        content = (out.content or "").strip()
        if not content:
            return None
    except Exception as e:
        print(f"    [요약 실패] {label}: {e}")
        return None
    meta = dict(doc_list[0].metadata or {})
    meta["chunk_kind"] = "summary"
    return Document(page_content=content, metadata=meta)


def _merge_summary_and_body_chunks(
    source_order: list[str],
    summary_by_source: dict[str, Document],
    body_chunks: list[Document],
) -> list[Document]:
    body_by: dict[str, list] = defaultdict(list)
    for ch in body_chunks:
        body_by[_doc_source_key(ch)].append(ch)
    merged: list[Document] = []
    for sk in source_order:
        if sk in summary_by_source:
            merged.append(summary_by_source[sk])
        merged.extend(body_by.get(sk, []))
    return merged


def load_documents_local(origins_cfg: dict):
    """data/portfolio/ 및 하위 폴더에서 PDF, DOCX, Markdown(.md) 로드. 출처 유형은 portfolio_origins.yaml."""
    docs = []
    portfolio_dir = config.PORTFOLIO_DIR
    if not portfolio_dir.exists():
        raise FileNotFoundError(f"포트폴리오 폴더가 없습니다: {portfolio_dir}")

    word_files = []
    pdf_files = []
    md_files = []
    excluded_files: list[str] = []
    for path in sorted(portfolio_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name == ".gitkeep":
            continue
        rel = path.relative_to(portfolio_dir)
        origin = resolve_portfolio_origin_local(portfolio_dir, rel, origins_cfg)
        if origin == "excluded":
            excluded_files.append(str(rel))
            continue
        extra = metadata_for_origin(origin)
        if path.suffix.lower() == ".pdf":
            pdf_files.append(str(rel))
            loader = PyPDFLoader(str(path))
            loaded = loader.load()
            _merge_doc_metadata(loaded, extra)
            docs.extend(loaded)
        elif path.suffix.lower() in (".docx", ".doc"):
            word_files.append(str(rel))
            loader = Docx2txtLoader(str(path))
            loaded = loader.load()
            _merge_doc_metadata(loaded, extra)
            docs.extend(loaded)
        elif path.suffix.lower() == ".md":
            md_files.append(str(rel))
            text = path.read_text(encoding="utf-8", errors="replace")
            md = {"source": str(path)}
            md.update(extra)
            docs.append(Document(page_content=text, metadata=md))

    print("  [로컬] 읽은 PDF:", pdf_files if pdf_files else "(없음)")
    print("  [로컬] 읽은 Word(.doc/.docx):", word_files if word_files else "(없음)")
    print("  [로컬] 읽은 Markdown:", md_files if md_files else "(없음)")
    if excluded_files:
        print(f"  [출처 excluded] 인덱스에서 제외 ({len(excluded_files)}개):")
        for e in excluded_files[:30]:
            print(f"    - {e}")
        if len(excluded_files) > 30:
            print(f"    ... 외 {len(excluded_files) - 30}개")
    if not docs:
        raise ValueError(
            f"문서가 없습니다. {portfolio_dir} 에 넣은 파일이 모두 portfolio_origins 에서 excluded 되었거나, "
            "인덱싱할 PDF/DOCX/.md 가 없습니다. data/portfolio_origins.yaml 의 default·rules 를 확인하세요."
        )
    return docs


def load_documents_from_drive(origins_cfg: dict):
    """구글 드라이브 폴더에서 문서 로드 (Google Docs, 스프레드시트, PDF 등)."""
    folder_id = config.GOOGLE_DRIVE_FOLDER_ID
    if not folder_id:
        raise ValueError(
            "GOOGLE_DRIVE_FOLDER_ID 가 비어 있습니다. "
            ".env 에 구글 드라이브 폴더 ID를 넣거나, 로컬 data/portfolio/ 를 사용하세요."
        )
    try:
        from langchain_google_community import GoogleDriveLoader
    except ImportError:
        raise ImportError(
            "구글 드라이브 로더를 쓰려면: uv add langchain-google-community[drive] "
            "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
    # drive.file 은 '앱이 연 파일'만 보임. 폴더 안 기존 문서를 보려면 drive.readonly 필요.
    # document=Google Docs, sheet=Google 스프레드시트, pdf=PDF (업로드된 .docx는 Google Docs로 변환된 경우만 해당)
    file_types = ["document", "sheet", "pdf"]
    print(f"  [구글 드라이브] 로드 대상 file_types: {file_types} (document=Google Docs, sheet=스프레드시트, pdf=PDF)")
    kwargs = {
        "folder_id": folder_id,
        "recursive": config.GOOGLE_DRIVE_RECURSIVE,
        "file_types": file_types,
        "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
    }
    if config.GOOGLE_DRIVE_CREDENTIALS_PATH:
        kwargs["credentials_path"] = config.GOOGLE_DRIVE_CREDENTIALS_PATH
    if config.GOOGLE_DRIVE_TOKEN_PATH:
        kwargs["token_path"] = config.GOOGLE_DRIVE_TOKEN_PATH
    loader = GoogleDriveLoader(**kwargs)
    docs = loader.load()
    if not docs:
        raise ValueError(
            f"구글 드라이브 폴더에서 문서를 찾지 못했습니다. folder_id={folder_id} 를 확인하고, "
            "폴더에 Google Docs/스프레드시트/PDF가 있는지 확인하세요."
        )
    # 로드된 파일 목록을 콘솔에 출력 (metadata.source 또는 파일명)
    seen = set()
    file_list = []
    for d in docs:
        src = (d.metadata.get("source") or d.metadata.get("title") or "").strip()
        if src and src not in seen:
            seen.add(src)
            file_list.append(src)
    print("  [구글 드라이브] 읽은 파일 목록 (문서별 1회씩):")
    for i, name in enumerate(file_list, 1):
        print(f"    {i}. {name}")
    print(f"  [구글 드라이브] 총 {len(file_list)}개 파일, {len(docs)}개 페이지/문서 단위")
    kept: list = []
    excluded_drive = 0
    for d in docs:
        src = (d.metadata.get("source") or d.metadata.get("title") or "").strip()
        title = (d.metadata.get("title") or "").strip()
        origin = resolve_portfolio_origin_drive(src, title, origins_cfg)
        if origin == "excluded":
            excluded_drive += 1
            continue
        md = dict(d.metadata or {})
        md.update(metadata_for_origin(origin))
        d.metadata = md
        kept.append(d)
    if excluded_drive:
        print(f"  [출처 excluded] 드라이브 문서 {excluded_drive}개 페이지/단위 는 인덱스에서 제외")
    if not kept:
        raise ValueError(
            "구글 드라이브에서 로드한 문서가 모두 portfolio_origins 에서 excluded 입니다. "
            "source_match·default·rules 를 확인하세요."
        )
    return kept


def main():
    use_drive = bool(config.GOOGLE_DRIVE_FOLDER_ID) and not getattr(
        config, "INDEX_BUILD_USE_LOCAL_ONLY", False
    )
    if config.GOOGLE_DRIVE_FOLDER_ID and getattr(config, "INDEX_BUILD_USE_LOCAL_ONLY", True):
        print(
            "  [인덱스 빌드] 로컬 우선 기본값: GOOGLE_DRIVE_FOLDER_ID 가 있어도 data/portfolio/ 만 사용합니다. "
            "(드라이브 빌드: .env 에 INDEX_BUILD_USE_LOCAL_ONLY=false)"
        )
    origins_path = getattr(config, "PORTFOLIO_ORIGINS_PATH", ROOT / "data" / "portfolio_origins.yaml")
    origins_cfg = load_portfolio_origins_config(origins_path)
    if origins_path.exists():
        print(f"  [출처 유형] {origins_path.name} 사용 (규칙 {len(origins_cfg.get('rules') or [])}개)")
    else:
        print(f"  [출처 유형] {origins_path.name} 없음 → 전부 default({origins_cfg.get('default', 'unspecified')})")

    if use_drive:
        # OAuth(credentials/token) 사용 시 서비스 계정이 우선되지 않도록 비움
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
        print("문서 로드 중 (구글 드라이브)...")
        raw_docs = load_documents_from_drive(origins_cfg)
    else:
        print("문서 로드 중 (로컬 data/portfolio)...")
        raw_docs = load_documents_local(origins_cfg)
    print(f"  로드된 페이지/문서 수: {len(raw_docs)}")

    source_order, by_source = _group_docs_by_source_ordered(raw_docs)

    print("청킹 중 (본문, chunk_kind=body)...")
    # 마크다운 ## / ### 단위로 먼저 나누어 섹션 단위 청크 유지 (문서 구조 보존)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    body_chunks = splitter.split_documents(raw_docs)
    for doc in body_chunks:
        md = dict(doc.metadata or {})
        md["chunk_kind"] = "body"
        doc.metadata = md

    summary_by_source: dict[str, Document] = {}
    if getattr(config, "INDEX_SUMMARY_ENABLED", False):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("  [요약] OPENAI_API_KEY 없음 → 요약 청크 없이 본문만 인덱싱")
        else:
            print(f"  파일별 요약 생성 중 (모델: {getattr(config, 'INDEX_SUMMARY_MODEL', 'gpt-4o-mini')})...")
            for sk in source_order:
                doc_list = by_source[sk]
                summ = _build_summary_document(sk, doc_list)
                if summ is not None:
                    summary_by_source[sk] = summ
                    try:
                        _shown = sk if sk == "__missing__" else Path(sk).name
                    except Exception:
                        _shown = sk
                    print(f"    [ok] 요약: {_shown}")
            print(f"  요약 청크 수: {len(summary_by_source)} / 소스 수: {len(source_order)}")
    else:
        print("  [요약] INDEX_SUMMARY_ENABLED=false → 요약 청크 생략")

    chunks = _merge_summary_and_body_chunks(source_order, summary_by_source, body_chunks)
    print(f"  최종 청크 수(요약+본문): {len(chunks)}")

    print("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    print("FAISS 인덱스 생성 중...")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(config.INDEX_DIR))
    print(f"  FAISS 저장 완료: {config.INDEX_DIR}")

    # BM25 인덱스 (Kiwi 형태소): HF 업로드 시 index/ 에 함께 포함
    print("BM25(Kiwi) 인덱스 생성 중...")
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    corpus_tokens = []
    for doc in chunks:
        tokens = [t.form for t in kiwi.tokenize(doc.page_content)]
        corpus_tokens.append(tokens)
    bm25 = BM25Okapi(corpus_tokens)
    with open(config.INDEX_DIR / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(corpus_tokens, f)
    with open(config.INDEX_DIR / "bm25_docs.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"  BM25 저장 완료: bm25_corpus.pkl, bm25_docs.pkl")


if __name__ == "__main__":
    main()
