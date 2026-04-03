"""포트폴리오 문서 출처(개인/회사/미표시/인덱스제외) — data/portfolio_origins.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

# excluded: 인덱스·RAG에 넣지 않음 (whitelist: rules/default로 명시된 파일만 사용)
VALID_ORIGINS = frozenset({"personal", "company", "unspecified", "excluded"})
INDEXABLE_ORIGINS = frozenset({"personal", "company", "unspecified"})

DEFAULT_CONFIG: dict = {"default": "unspecified", "rules": [], "source_match": []}


def load_portfolio_origins_config(path: Path | None) -> dict:
    """YAML 없거나 깨지면 DEFAULT_CONFIG."""
    if path is None or not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return dict(DEFAULT_CONFIG)
    default = raw.get("default", "unspecified")
    if default not in VALID_ORIGINS:
        default = "unspecified"
    rules = raw.get("rules") or []
    if not isinstance(rules, list):
        rules = []
    source_match = raw.get("source_match") or []
    if not isinstance(source_match, list):
        source_match = []
    return {"default": default, "rules": rules, "source_match": source_match}


def resolve_portfolio_origin_local(portfolio_dir: Path, rel: Path, cfg: dict) -> str:
    """data/portfolio 기준 상대 경로 rel에 대해 rules[].glob 첫 매칭 (위→아래 순)."""
    default = cfg.get("default", "unspecified")
    if default not in VALID_ORIGINS:
        default = "unspecified"
    rules = cfg.get("rules") or []
    try:
        root = portfolio_dir.resolve()
        abs_file = (portfolio_dir / rel).resolve()
        abs_file.relative_to(root)
    except Exception:
        return default
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        pat = (rule.get("glob") or "").strip()
        origin = rule.get("origin", "unspecified")
        if origin not in VALID_ORIGINS or not pat:
            continue
        try:
            for hit in portfolio_dir.glob(pat):
                try:
                    if hit.resolve() == abs_file:
                        return origin
                except OSError:
                    if hit == abs_file:
                        return origin
        except Exception:
            continue
    return default


def resolve_portfolio_origin_drive(source: str, title: str, cfg: dict) -> str:
    """드라이브 등 로컬 상대경로가 없을 때: source_match[].contains 대소문자 무시 부분일치 (순서대로)."""
    default = cfg.get("default", "unspecified")
    if default not in VALID_ORIGINS:
        default = "unspecified"
    hay = f"{source or ''} {title or ''}".lower()
    for rule in cfg.get("source_match") or []:
        if not isinstance(rule, dict):
            continue
        sub = (rule.get("contains") or "").strip().lower()
        origin = rule.get("origin", "unspecified")
        if not sub or origin not in VALID_ORIGINS:
            continue
        if sub in hay:
            return origin
    return default


def metadata_for_origin(origin: str) -> dict:
    """인덱싱 대상 문서만. excluded 는 빌드 단계에서 로드하지 않음."""
    if origin not in INDEXABLE_ORIGINS:
        return {}
    return {"portfolio_origin": origin}
