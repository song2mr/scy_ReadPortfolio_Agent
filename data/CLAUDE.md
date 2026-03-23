# data/ — 포트폴리오 문서 & 후보자 정보

## 포트폴리오 문서 (`portfolio/`)

이력서·경력기술서·프로젝트 설명서 등 PDF/DOCX 파일을 여기에 배치.
변경 후 `uv run python scripts/build_index.py` 재실행 필요.

## 후보자 메타 정보 (`candidate_profile.py`)

- `PROFILE_BASIC` — 시스템 프롬프트에 주입되는 기본 경력·학력·강점 요약
- `QUERY_EXPANSION_TOPICS` — 쿼리 확장 시 LLM에 전달하는 프로젝트·솔루션 목록

후보자 변경 시 이 파일의 두 상수를 수정 (+ 포트폴리오 문서 교체 + 인덱스 재빌드).
