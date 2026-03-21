"""
포트폴리오 문서를 로드·청킹·임베딩 후 FAISS + BM25(Kiwi) 인덱스를 index/ 에 저장합니다.
HF 업로드 시 index/ 폴더 전체(index.faiss, index.pkl, bm25_*.pkl)를 올리면 됩니다.
- 로컬: data/portfolio/ 의 PDF, DOCX 사용
- 구글 드라이브: .env 에 GOOGLE_DRIVE_FOLDER_ID 설정 시 해당 폴더에서 문서 로드
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

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

import config


def load_documents_local():
    """data/portfolio/ 에서 PDF, DOCX 로드."""
    docs = []
    portfolio_dir = config.PORTFOLIO_DIR
    if not portfolio_dir.exists():
        raise FileNotFoundError(f"포트폴리오 폴더가 없습니다: {portfolio_dir}")

    word_files = []
    pdf_files = []
    for path in sorted(portfolio_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            pdf_files.append(path.name)
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix.lower() in (".docx", ".doc"):
            word_files.append(path.name)
            loader = Docx2txtLoader(str(path))
            docs.extend(loader.load())

    print("  [로컬] 읽은 PDF:", pdf_files if pdf_files else "(없음)")
    print("  [로컬] 읽은 Word(.doc/.docx):", word_files if word_files else "(없음)")
    if not docs:
        raise ValueError(f"문서가 없습니다. {portfolio_dir} 에 PDF 또는 DOCX를 넣어주세요.")
    return docs


def load_documents_from_drive():
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
    return docs


def main():
    use_drive = bool(config.GOOGLE_DRIVE_FOLDER_ID)
    if use_drive:
        # OAuth(credentials/token) 사용 시 서비스 계정이 우선되지 않도록 비움
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
        print("문서 로드 중 (구글 드라이브)...")
        raw_docs = load_documents_from_drive()
    else:
        print("문서 로드 중 (로컬 data/portfolio)...")
        raw_docs = load_documents_local()
    print(f"  로드된 페이지/문서 수: {len(raw_docs)}")

    print("청킹 중...")
    # 마크다운 ## / ### 단위로 먼저 나누어 섹션 단위 청크 유지 (문서 구조 보존)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  청크 수: {len(chunks)}")

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
