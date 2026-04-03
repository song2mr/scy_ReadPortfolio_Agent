"""
Hugging Face Spaces / Gradio 진입점.
로컬: uv run python app.py  또는  uv run python -m app.app
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# .env 로드 (루트에서 실행해도 프로젝트 루트의 .env 사용. Space에서는 없음)
load_dotenv(Path(__file__).resolve().parent / ".env")

# LangSmith: HF Space 등에서 env만으로 안 될 때 명시적으로 설정. 반드시 langchain import 전에 실행.
_api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
_tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")
if _api_key and _tracing:
    try:
        from langsmith import Client, configure
        configure(
            enabled=True,
            client=Client(api_key=_api_key),
            project_name=os.getenv("LANGSMITH_PROJECT", "").strip() or None,
        )
    except Exception:
        pass  # 실패 시 env만 의존하고 진행

import gradio as gr

from app.app import build_ui

demo = build_ui()

if __name__ == "__main__":
    theme = gr.themes.Glass(primary_hue="indigo", secondary_hue="slate")
    css = """
    .stats-bar { padding: 0.6rem 1rem; border-radius: 10px; background: var(--block-background-fill); border: 1px solid var(--block-border-color); }
    .chat-container { min-height: 420px; border-radius: 12px !important; }
    .app-hero { padding: 1.1rem 1.35rem; border-radius: 16px; margin-bottom: 0.75rem;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.14), rgba(14, 165, 233, 0.08));
      border: 1px solid rgba(99, 102, 241, 0.28); box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06); }
    .app-hero-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
    .app-hero-sub { font-size: 0.92rem; opacity: 0.88; line-height: 1.45; }
    .main-tabs .tab-nav button { font-weight: 600 !important; border-radius: 10px 10px 0 0 !important; }
    .panel-card { padding: 1rem 1.15rem; border-radius: 14px; border: 1px solid var(--block-border-color);
      background: var(--block-background-fill); margin-bottom: 1rem; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04); }
    .panel-card-accent { border-color: rgba(99, 102, 241, 0.35);
      background: linear-gradient(180deg, rgba(99, 102, 241, 0.06), var(--block-background-fill)); }
    .panel-heading { margin: 0 0 0.5rem 0; font-size: 1.08rem; font-weight: 650; }
    .panel-desc { margin: 0; font-size: 0.9rem; line-height: 1.55; opacity: 0.92; }
    .cta-button { min-height: 2.75rem !important; font-weight: 600 !important; border-radius: 10px !important; }
    .job-input textarea, .job-input input { border-radius: 10px !important; }
    .insight-output { padding: 1rem 1.15rem; border-radius: 12px; border: 1px solid var(--block-border-color);
      background: var(--block-background-fill); min-height: 120px; }
    .prompt-preview { font-size: 0.82rem !important; max-height: 320px; overflow: auto; }
    .gradio-container .block.examples, .gradio-container [class*="examples"] { display: none !important; }
    """
    demo.launch(theme=theme, css=css)
