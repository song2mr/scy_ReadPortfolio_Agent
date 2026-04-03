"""
data/portfolio 를 스캔해, 어떤 규칙에도 안 걸리는 파일을
data/portfolio_origins.yaml 의 rules 에 자동으로 추가합니다 (origin: unspecified).

실행 (프로젝트 루트):
  uv run python scripts/portfolio_origins_audit.py          # yaml 갱신 + 요약 출력
  uv run python scripts/portfolio_origins_audit.py --dry-run   # 출력만, 파일 안 씀
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import config
from app.portfolio_origins import (
    VALID_ORIGINS,
    load_portfolio_origins_config,
    resolve_portfolio_origin_local,
)


def _iter_portfolio_files(portfolio_dir: Path):
    for path in sorted(portfolio_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name == ".gitkeep":
            continue
        suf = path.suffix.lower()
        if suf not in (".pdf", ".docx", ".doc", ".md"):
            continue
        yield path


def _normalize_rules(raw) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict) and (item.get("glob") or "").strip():
            o = item.get("origin", "unspecified")
            if o not in VALID_ORIGINS:
                o = "unspecified"
            out.append({"glob": str(item["glob"]).strip(), "origin": o})
    return out


def _first_matching_rule_origin(portfolio_dir: Path, rel: Path, rules: list[dict]) -> str | None:
    """이 파일에 매칭되는 첫 번째 rule 의 origin. 매칭 없으면 None (default 로 떨어짐)."""
    try:
        abs_file = (portfolio_dir / rel).resolve()
        abs_file.relative_to(portfolio_dir.resolve())
    except Exception:
        return None
    for rule in rules:
        pat = rule.get("glob") or ""
        if not pat:
            continue
        try:
            for hit in portfolio_dir.glob(pat):
                try:
                    if hit.resolve() == abs_file:
                        return rule.get("origin", "unspecified")
                except OSError:
                    if hit == abs_file:
                        return rule.get("origin", "unspecified")
        except Exception:
            continue
    return None


def _read_yaml_split(path: Path) -> tuple[str, dict]:
    """default: 이전은 헤더(주석)로 보존, 이후는 YAML 로 파싱."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("default:"):
            idx = i
            break
    if idx is None:
        raise SystemExit(f"{path}: 'default:' 로 시작하는 줄이 필요합니다.")
    header = "\n".join(lines[:idx]).rstrip()
    body_text = "\n".join(lines[idx:])
    data = yaml.safe_load(body_text) or {}
    if not isinstance(data, dict):
        data = {}
    return header + "\n", data


def _write_yaml_split(path: Path, header: str, data: dict) -> None:
    body = yaml.dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    path.write_text(header.rstrip() + "\n" + body, encoding="utf-8")


def _collect_to_add(portfolio_dir: Path, rules: list[dict], default: str) -> list[dict]:
    """어떤 rule 에도 매칭되지 않는 파일마다 glob 고정 + unspecified 행 추가. default=excluded 면 비움(자동 화이트리스트 안 함)."""
    if default == "excluded":
        return []
    to_add: list[dict] = []
    seen_glob: set[str] = set()
    for path in _iter_portfolio_files(portfolio_dir):
        rel = path.relative_to(portfolio_dir)
        if _first_matching_rule_origin(portfolio_dir, rel, rules) is not None:
            continue
        g = rel.as_posix()
        if g in seen_glob:
            continue
        seen_glob.add(g)
        to_add.append({"glob": g, "origin": "unspecified"})
    return to_add


def main() -> None:
    ap = argparse.ArgumentParser(description="portfolio_origins.yaml 에 미매칭 파일 규칙 자동 추가")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="요약만 출력하고 yaml 은 수정하지 않음",
    )
    args = ap.parse_args()

    portfolio_dir = config.PORTFOLIO_DIR
    origins_path = getattr(config, "PORTFOLIO_ORIGINS_PATH", ROOT / "data" / "portfolio_origins.yaml")

    if not portfolio_dir.is_dir():
        print(f"폴더 없음: {portfolio_dir}")
        sys.exit(1)
    if not origins_path.is_file():
        print(f"파일 없음: {origins_path}")
        sys.exit(1)

    header, data = _read_yaml_split(origins_path)
    rules = _normalize_rules(data.get("rules"))
    data["rules"] = rules
    data.setdefault("default", "unspecified")
    data.setdefault("source_match", [])

    cfg = load_portfolio_origins_config(origins_path)
    rows: list[tuple[str, str]] = []
    for path in _iter_portfolio_files(portfolio_dir):
        rel = path.relative_to(portfolio_dir)
        origin = resolve_portfolio_origin_local(portfolio_dir, rel, cfg)
        rows.append((rel.as_posix(), origin))

    by_origin: dict[str, list[str]] = {}
    for rel, o in rows:
        by_origin.setdefault(o, []).append(rel)

    to_add = _collect_to_add(portfolio_dir, rules, cfg.get("default", "unspecified"))

    print(f"설정 파일: {origins_path}", flush=True)
    print(f"스캔 경로: {portfolio_dir}", flush=True)
    print(f"default: {cfg.get('default', 'unspecified')!r}", flush=True)
    print(f"기존 규칙 수: {len(rules)}, audit 추가 예정: {len(to_add)}개\n", flush=True)

    for label, key in (
        ("개인 (personal)", "personal"),
        ("회사/기관 (company)", "company"),
        ("미분류 (unspecified)", "unspecified"),
        ("인덱스 제외 (excluded)", "excluded"),
    ):
        files = sorted(by_origin.get(key, []))
        print(f"=== {label}: {len(files)}개 ===", flush=True)
        for r in files:
            print(f"  - {r}", flush=True)
        print(flush=True)

    if not to_add:
        if cfg.get("default") == "excluded":
            print(
                "(default=excluded: 규칙 없는 파일은 인덱스에 넣지 않습니다. "
                "audit는 glob를 자동 추가하지 않으니, 넣을 파일만 rules에 직접 추가하세요.)",
                flush=True,
            )
        else:
            print("(추가할 규칙 없음. 모든 파일이 기존 rules 에 매칭됩니다.)", flush=True)
        return

    print("--- yaml 에 추가할 규칙 (origin 은 파일에서 personal/company 로 고치면 됨) ---", flush=True)
    for item in to_add:
        print(f"  - glob: {item['glob']!r}", flush=True)
        print(f"    origin: {item['origin']}", flush=True)
    print(flush=True)

    if args.dry_run:
        print("(dry-run: 파일을 수정하지 않았습니다.)", flush=True)
        return

    out_data = {
        "default": data.get("default", "unspecified"),
        "rules": rules + to_add,
        "source_match": list(data.get("source_match") or []),
    }
    _write_yaml_split(origins_path, header, out_data)
    print(f"갱신 완료: {len(to_add)}개 규칙을 rules 끝에 추가했습니다. 인덱스: uv run python scripts/build_index.py", flush=True)


if __name__ == "__main__":
    main()
