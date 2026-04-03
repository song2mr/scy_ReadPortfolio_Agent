# data/ — 포트폴리오 문서·후보자 정보·출처 분류

## 포트폴리오 문서 (`portfolio/`)

이력서·경력기술서·프로젝트 설명서 등 **PDF / DOCX / Markdown(.md)** 를 배치 (하위 폴더 포함 스캔).
변경 후 `uv run python scripts/build_index.py` 재실행.

## 출처 분류 (`portfolio_origins.yaml`)

- 빌드 시 각 파일에 **`portfolio_origin`** (`personal` | `company` | `unspecified`) 를 메타로 박음. **`excluded`** 이면 그 파일은 **인덱스에 넣지 않음** (화이트리스트).
- `rules`: `data/portfolio` 기준 **glob**, 위→아래 **첫 매칭**만 적용. `default`는 어디에도 안 걸릴 때.
- **채팅 시 yaml을 읽지 않음** — 반드시 재빌드 후 `index/`·`bm25_docs.pkl` 에 반영된 메타가 `app/rag.py`의 `_format_docs`로 LLM에 라벨 전달.
- 자동 점검·미매칭 파일 yaml 반영:  
  `uv run python scripts/portfolio_origins_audit.py` (`--dry-run` 미리보기). `default: excluded` 일 때는 규칙 자동 대량 추가 안 함.

## 후보자 메타 (`candidate_profile.py`)

- `PROFILE_BASIC` — 시스템 프롬프트 기본 경력·학력·강점
- `QUERY_EXPANSION_TOPICS` — 쿼리 확장 시 LLM에 넘기는 프로젝트·솔루션 힌트 목록

후보자 변경 시 두 상수 수정 + 포트폴리오 교체 + 인덱스 재빌드.
