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
    .stats-bar { padding: 0.6rem 1rem; border-radius: 8px; } .chat-container { min-height: 420px; }
    .gradio-container .block.examples, .gradio-container [class*="examples"] { display: none !important; }
    """
    demo.launch(theme=theme, css=css)
