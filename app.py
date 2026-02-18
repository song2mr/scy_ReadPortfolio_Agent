"""
Hugging Face Spaces / Gradio 진입점.
로컬: uv run python app.py  또는  uv run python -m app.app
"""
from pathlib import Path

from dotenv import load_dotenv

# .env 로드 (루트에서 실행해도 프로젝트 루트의 .env 사용)
load_dotenv(Path(__file__).resolve().parent / ".env")

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
