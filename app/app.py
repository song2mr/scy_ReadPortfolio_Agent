
"""
포트폴리오 RAG Gradio 앱 — 채팅, 추천 질문, 대화 초기화, 요약(MD/PDF) 다운로드, 키워드 통계, 스트리밍.
실행: 프로젝트 루트에서 uv run python -m app.app
"""
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# .env는 프로젝트 루트에서 명시적으로 로드 (실행 경로와 무관하게)
load_dotenv(ROOT / ".env")

import gradio as gr
from langchain_openai import ChatOpenAI

import config

from app.rag import (
    evaluate_job_fit_for_role,
    generate_intro_from_all_summaries,
    get_answer,
    get_answer_stream,
    get_intro_prompt_placeholder_display,
    get_job_fit_prompt_placeholder_display,
)

APP_VERSION = config.APP_VERSION

# 첫 인사 메시지 — 인사 담당자에게 송찬영을 소개하는 에이전트임을 명시
FIRST_MESSAGE = """안녕하세요. 저는 **송찬영**을 인사·채용 담당자께 소개하는 에이전트입니다.
송찬영은 데이터 분석가이자 LLM 에이전트 개발·설계자입니다. 포트폴리오(RAG)를 바탕으로 궁금하신 점에 답해 드립니다.
경력, 강점, 프로젝트 등 무엇이든 질문해 주세요."""

# 프리셋 질문 (MEMO: 2~3개) — gr.Examples에도 사용
PRESET_QUESTIONS = [
    "이 사람의 강점을 한 줄로 요약해 주세요.",
    "주요 경력과 프로젝트를 간단히 소개해 주세요.",
    "어떤 역할/포지션에 적합해 보이나요?",
]

# 키워드 통계용 (질문 주제)
KEYWORDS = ["경력", "강점", "프로젝트", "역할", "경험", "역량", "스킬", "학력"]

# Gradio 6 Chatbot: content 는 문자열. (리스트 [["text", "...]] 형식은 초기 value 에서 오류 남)
def _to_messages(pairs: list) -> list:
    """pairs [[user,bot],...] -> Gradio 6 messages [{role, content},...] (user 먼저, assistant 다음)"""
    out = []
    for user, bot in pairs:
        if user is not None and str(user).strip():
            out.append({"role": "user", "content": str(user).strip()})
        if bot is not None and str(bot).strip():
            out.append({"role": "assistant", "content": str(bot).strip()})
    return out


def _content_to_str(content) -> str:
    """Gradio 6 에서 content 가 리스트/딕셔너리로 올 수 있음 → 항상 문자열로."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", str(content))))
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] == "text":
                parts.append(str(x[1]))
            elif isinstance(x, dict):
                parts.append(x.get("text", x.get("content", str(x))))
            else:
                parts.append(str(x))
        return " ".join(parts)
    return str(content)


def _from_messages(messages: list) -> list:
    """Gradio 6 messages -> pairs [[user,bot],...]"""
    if not messages:
        return []
    pairs = []
    current_user = None
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        content = _content_to_str(content)
        if role == "assistant":
            pairs.append([current_user, content or ""])
            current_user = None
        elif role == "user":
            current_user = content or ""
    return pairs


# 대화량 등 통계용
def _strip_html(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _stats_from_history(history: list) -> tuple[int, int]:
    """히스토리에서 턴 수(사용자 질문이 있는 Q&A 수), 총 글자 수 반환."""
    if not history:
        return 0, 0
    turns = sum(1 for user, _ in history if (user or "").strip())
    total_chars = sum(
        len(_strip_html(user or "")) + len(_strip_html(bot or ""))
        for user, bot in history
    )
    return turns, total_chars


def _format_stats(turns: int, chars: int) -> str:
    return f"💬 대화 **{turns}**턴 · **{chars:,}**자"


def _ref_heading(doc) -> str:
    md = getattr(doc, "metadata", None) or {}
    src = (md.get("source") or md.get("title") or "").strip()
    origin = md.get("portfolio_origin")
    if not origin and md.get("personal_project"):
        origin = "personal"
    if origin == "personal":
        tag = " · 개인 프로젝트"
    elif origin == "company":
        tag = " · 회사/기관"
    else:
        tag = ""
    if src:
        try:
            name = Path(src).name
        except Exception:
            name = src[-60:]
        return f"**{name}**{tag}"
    return f"참고{tag}" if tag else "참고"


def _format_response(answer: str, source_docs) -> str:
    """답변 + 참고 문단(접기) 포맷."""
    if not source_docs:
        return answer
    blocks = []
    for d in source_docs:
        head = _ref_heading(d)
        body = d.page_content.strip()
        snippet = body[:400] + "..." if len(body) > 400 else body
        blocks.append(f"• {head}\n\n{snippet}")
    refs = "\n\n".join(blocks)
    return answer + "\n\n<details><summary>📎 참고한 문단</summary>\n\n" + refs + "\n</details>"


def _submit(message: str, history: list) -> tuple[list, str, str, str]:
    """전송 시 RAG 답변 반환, 히스토리·통계·키워드 갱신. (history, clear_txt, stats_md, keyword_md)."""
    if not message or not message.strip():
        return history, "", _format_stats(*_stats_from_history(history)), _keyword_stats(history)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        h = history + [[message, "⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요."]]
        return h, "", _format_stats(*_stats_from_history(h)), _keyword_stats(h)
    try:
        answer, source_docs = get_answer(message.strip(), history_pairs=history)
        full = _format_response(answer, source_docs) if source_docs else answer
        new_history = history + [[message, full]]
        return new_history, "", _format_stats(*_stats_from_history(new_history)), _keyword_stats(new_history)
    except Exception as e:
        new_history = history + [[message, f"⚠️ 오류가 발생했습니다: {str(e)}"]]
        return new_history, "", _format_stats(*_stats_from_history(new_history)), _keyword_stats(new_history)


def _submit_stream(message: str, history: list):
    """스트리밍 전송: (history, clear_txt, stats_md, keyword_md) 를 여러 번 yield."""
    if not message or not message.strip():
        yield history, "", _format_stats(*_stats_from_history(history)), _keyword_stats(history)
        return
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        h = history + [[message, "⚠️ OPENAI_API_KEY가 설정되지 않았습니다."]]
        yield h, "", _format_stats(*_stats_from_history(h)), _keyword_stats(h)
        return
    # 인덱스 여부는 rag에서 처리 (포폴 무관 질문은 인덱스 없어도 LLM만으로 답변)
    try:
        if os.getenv("DEBUG_RAG", "").strip() in ("1", "true", "yes"):
            print("[APP] get_answer_stream 호출 직전", flush=True)
        last_full, last_sources = None, None
        for full, source_docs in get_answer_stream(message.strip(), history_pairs=history):
            last_full, last_sources = full, source_docs
            new_h = history + [[message, full]]
            yield new_h, "", _format_stats(*_stats_from_history(new_h)), _keyword_stats(new_h)
        if last_full is not None and last_sources is not None:
            final = _format_response(last_full, last_sources) if last_sources else last_full
            new_h = history + [[message, final]]
            yield new_h, "", _format_stats(*_stats_from_history(new_h)), _keyword_stats(new_h)
    except Exception as e:
        h = history + [[message, f"⚠️ 오류: {str(e)}"]]
        yield h, "", _format_stats(*_stats_from_history(h)), _keyword_stats(h)


def _build_transcript(history: list) -> str:
    """히스토리를 '질문 / 답변' 텍스트로 변환 (HTML 제거)."""
    lines = []
    for user, bot in history:
        u = _strip_html(user or "")
        b = _strip_html(bot or "")
        if u:
            lines.append(f"### 질문\n{u}")
        if b:
            lines.append(f"### 답변\n{b}\n")
    return "\n".join(lines)


_SUMMARY_PROMPT = "다음 포트폴리오 Q&A 대화를 마크다운을 적절히 사용해서 가독성 있게 요약해 주세요. 지원자의 이력서, 포트폴리오를 작성한다고 가정하세요. 한국어로만 작성하고, 지원자 관점이 아니라 인사 담당자가 읽을 요약으로 작성하세요.\n\n"


def _generate_summary_text(transcript: str) -> str:
    """대화 전사를 LLM으로 요약. API 키 없거나 실패 시 기본 메시지 반환."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "대화 내용이 없거나 요약할 수 없습니다."
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        return llm.invoke(_SUMMARY_PROMPT + transcript[:8000]).content
    except Exception:
        return "(요약 생성 중 오류가 발생했습니다.)"


def _generate_summary_file(history: list) -> str | None:
    """대화 요약 + 전체 내역을 마크다운 파일로 저장하고 경로 반환. 다운로드용."""
    if not history or _stats_from_history(history) == (0, 0):
        return None
    transcript = _build_transcript(history)
    if not transcript.strip():
        return None
    summary = _generate_summary_text(transcript)
    content = f"""# 포트폴리오 대화 요약
생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 요약
{summary}

---
## 전체 대화
{transcript}
"""
    fd, path = tempfile.mkstemp(suffix=".md", prefix="portfolio_chat_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _keyword_stats(history: list) -> str:
    """질문에서 키워드 언급 횟수 → 마크다운 문자열."""
    if not history:
        return "### 📊 질문 키워드\n\n아직 질문이 없습니다."
    all_user = " ".join(_strip_html(user or "") for user, _ in history if (user or "").strip())
    if not all_user.strip():
        return "### 📊 질문 키워드\n\n아직 질문이 없습니다."
    counts = [(k, all_user.count(k)) for k in KEYWORDS if all_user.count(k) > 0]
    if not counts:
        return "### 📊 질문 키워드\n\n(언급된 키워드 없음)"
    lines = [f"- **{k}**: {'█' * c} ({c}회)" for k, c in sorted(counts, key=lambda x: -x[1])]
    return "### 📊 질문 키워드\n\n" + "\n".join(lines)


def _generate_summary_pdf_or_txt(history: list) -> tuple[str, str]:
    """대화 요약을 PDF로 저장 시도, 실패 시 .txt. (경로, 'pdf'|'txt') 반환."""
    if not history or _stats_from_history(history) == (0, 0):
        return None, ""
    transcript = _build_transcript(history)
    if not transcript.strip():
        return None, ""
    summary = _generate_summary_text(transcript)
    text_content = f"포트폴리오 대화 요약\n생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n## 요약\n{summary}\n\n---\n## 전체 대화\n{transcript}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_LEFT

        font_name = "Helvetica"
        if os.name == "nt":
            try:
                pdfmetrics.registerFont(TTFont("Malgun", "malgun.ttf"))
                font_name = "Malgun"
            except Exception:
                try:
                    pdfmetrics.registerFont(TTFont("Gulim", "gulim.ttc"))
                    font_name = "Gulim"
                except Exception:
                    pass
        else:
            for path in [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]:
                if os.path.isfile(path):
                    try:
                        name = "KoreanFont"
                        pdfmetrics.registerFont(TTFont(name, path))
                        font_name = name
                        break
                    except Exception:
                        pass
        path_pdf = os.path.join(tempfile.gettempdir(), f"portfolio_summary_{ts}.pdf")
        doc = SimpleDocTemplate(path_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("T", parent=styles["Heading1"], fontName=font_name, fontSize=16, spaceAfter=20, alignment=TA_LEFT)
        body_style = ParagraphStyle("B", parent=styles["BodyText"], fontName=font_name, fontSize=10, spaceAfter=10, leading=14)
        story = []
        story.append(Paragraph("포트폴리오 대화 요약", title_style))
        story.append(Paragraph(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("요약", body_style))
        safe = re.sub(r"[#*`]", "", summary).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.append(Paragraph(safe[:5000], body_style))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("전체 대화", body_style))
        safe_t = re.sub(r"[#*`]", "", transcript).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.append(Paragraph(safe_t[:15000], body_style))
        doc.build(story)
        return path_pdf, "pdf"
    except Exception:
        path_txt = os.path.join(tempfile.gettempdir(), f"portfolio_summary_{ts}.txt")
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(text_content)
        return path_txt, "txt"


def _version_footer_md() -> str:
    return f'<p style="margin:0;opacity:0.75;font-size:0.85rem;">앱 버전 <strong>v{APP_VERSION}</strong></p>'


def build_ui():
    with gr.Blocks(title=f"포트폴리오 Q&A v{APP_VERSION}") as demo:
        # 비밀번호 미설정 시 로컬용: "dev" 입력 시 입장. 배포 시 Space Secrets에 APP_PASSWORD 설정.
        _app_password = os.getenv("APP_PASSWORD", "").strip()
        unlocked = gr.State(False)

        with gr.Column(visible=True) as pwd_section:
            gr.Markdown("### 🔐 입장을 위해 비밀번호를 입력하세요")
            gr.Markdown(_version_footer_md())
            pwd_input = gr.Textbox(
                label="비밀번호",
                type="password",
                placeholder="전달받은 비밀번호 입력",
                max_lines=1,
            )
            pwd_btn = gr.Button("입장", variant="primary")
            pwd_error = gr.Markdown("", visible=False)

        with gr.Column(visible=False) as chat_section:
            gr.Markdown(
                f"""
<div class="app-hero">
  <div class="app-hero-title">포트폴리오 에이전트</div>
  <div class="app-hero-sub">RAG 검색 · 프로젝트 요약 기반 소개글 · 직무 적합성 평가</div>
  {_version_footer_md()}
</div>
"""
            )
            with gr.Tabs(elem_classes=["main-tabs"]) as _main_tabs:
                with gr.Tab("💬 포트폴리오 Q&A"):
                    stats_md = gr.Markdown(_format_stats(0, 0), elem_classes=["stats-bar"])
                    chatbot = gr.Chatbot(
                        value=_to_messages([[None, FIRST_MESSAGE]]),
                        height=420,
                        elem_classes=["chat-container"],
                        render_markdown=True,
                    )
                    with gr.Accordion("✨ 추천 질문", open=False):
                        with gr.Row():
                            preset_btns = [gr.Button(q, size="sm", variant="secondary") for q in PRESET_QUESTIONS]
                    txt = gr.Textbox(
                        placeholder="질문을 입력하세요... (Enter로 전송)",
                        label="",
                        scale=7,
                        container=False,
                        max_lines=2,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("전송", variant="primary", size="lg")
                        summary_btn = gr.Button("📥 요약 다운로드 (MD)", variant="secondary", size="lg")
                        summary_pdf_btn = gr.Button("📥 요약 다운로드 (PDF)", variant="secondary", size="lg")
                        clear_btn = gr.Button("🗑️ 대화 초기화", variant="stop", size="lg")
                    summary_file = gr.File(label="대화 요약 파일", visible=True, interactive=False)
                    keyword_md = gr.Markdown(_keyword_stats([]), elem_classes=["stats-bar"])
                    for btn, q in zip(preset_btns, PRESET_QUESTIONS):
                        btn.click(fn=lambda q=q: q, outputs=[txt])
                    LOADING_PLACEHOLDER = "⏳ 검색·답변 생성 중..."

                    def _submit_stream_ui(message, chat_value):
                        pairs = _from_messages(chat_value or [])
                        msg = (message or "").strip()
                        if not msg:
                            yield _to_messages(pairs), "", _format_stats(*_stats_from_history(pairs)), _keyword_stats(pairs)
                            return
                        initial_pairs = pairs + [[msg, LOADING_PLACEHOLDER]]
                        yield _to_messages(initial_pairs), "", _format_stats(*_stats_from_history(pairs)), _keyword_stats(pairs)
                        for new_pairs, clear_txt, stats, keywords in _submit_stream(msg, pairs):
                            yield _to_messages(new_pairs), clear_txt, stats, keywords

                    submit_btn.click(
                        fn=_submit_stream_ui,
                        inputs=[txt, chatbot],
                        outputs=[chatbot, txt, stats_md, keyword_md],
                        show_progress=False,
                    ).then(fn=lambda: "", outputs=[txt])
                    txt.submit(
                        fn=_submit_stream_ui,
                        inputs=[txt, chatbot],
                        outputs=[chatbot, txt, stats_md, keyword_md],
                        show_progress=False,
                    ).then(fn=lambda: "", outputs=[txt])

                    def on_clear():
                        gr.Info("대화가 초기화되었습니다.")
                        return _to_messages([[None, FIRST_MESSAGE]]), _format_stats(0, 0), _keyword_stats([])

                    clear_btn.click(
                        fn=on_clear,
                        inputs=[],
                        outputs=[chatbot, stats_md, keyword_md],
                    )

                    def on_summary_md(chat_value):
                        pairs = _from_messages(chat_value or [])
                        path = _generate_summary_file(pairs)
                        if path:
                            gr.Info("요약 파일이 준비되었습니다. 아래에서 다운로드하세요.")
                            return path
                        gr.Warning("대화 내용이 없습니다. 먼저 질문을 나눈 뒤 다시 시도해 주세요.")
                        return None

                    summary_btn.click(
                        fn=on_summary_md,
                        inputs=[chatbot],
                        outputs=[summary_file],
                    )

                    def on_summary_pdf(chat_value):
                        pairs = _from_messages(chat_value or [])
                        path, fmt = _generate_summary_pdf_or_txt(pairs)
                        if not path:
                            gr.Warning("대화 내용이 없습니다. 먼저 질문을 나눈 뒤 다시 시도해 주세요.")
                            return None
                        if fmt == "pdf":
                            gr.Info("PDF 파일이 준비되었습니다. 아래에서 다운로드하세요.")
                        else:
                            gr.Warning("한글 PDF 생성에 실패해 텍스트(.txt) 파일로 저장했습니다. 아래에서 다운로드하세요.")
                        return path

                    summary_pdf_btn.click(
                        fn=on_summary_pdf,
                        inputs=[chatbot],
                        outputs=[summary_file],
                    )

                with gr.Tab("✍️ 소개글 작성"):
                    gr.Markdown(
                        """
<div class="panel-card panel-card-accent">
<h3 class="panel-heading">인사 담당자용 소개글</h3>
<p class="panel-desc">인덱스에 빌드된 <strong>전체 프로젝트 요약</strong>(파일당 1개 요약 청크)과 기본 프로필을 읽고, 한 편의 소개글을 작성합니다. 채팅 히스토리와는 별도로 동작합니다.</p>
</div>
"""
                    )
                    intro_generate_btn = gr.Button(
                        "전체 프로젝트 요약 기반으로 소개글 작성",
                        variant="primary",
                        size="lg",
                        elem_classes=["cta-button"],
                    )
                    intro_output = gr.Markdown(elem_classes=["insight-output"])

                    def _run_intro():
                        if not os.getenv("OPENAI_API_KEY", "").strip():
                            return "⚠️ OPENAI_API_KEY가 필요합니다."
                        return generate_intro_from_all_summaries()

                    intro_generate_btn.click(
                        fn=_run_intro,
                        inputs=[],
                        outputs=[intro_output],
                    )
                    with gr.Accordion("🔍 LLM 시스템 지시문 (실제 호출 시 요약·프로필이 채워집니다)", open=False):
                        gr.Markdown(
                            f"```text\n{get_intro_prompt_placeholder_display()}\n```",
                            elem_classes=["prompt-preview"],
                        )

                with gr.Tab("🎯 직무 적합성"):
                    gr.Markdown(
                        """
<div class="panel-card">
<h3 class="panel-heading">직무 적합성 평가</h3>
<p class="panel-desc">같은 <strong>전체 프로젝트 요약 + 기본 프로필</strong>을 바탕으로, 입력하신 직무에 대한 적합성을 정리해 드립니다. 참고용이며 최종 판단은 채용 절차에서 이루어져야 합니다.</p>
</div>
"""
                    )
                    job_title_in = gr.Textbox(
                        label="평가할 직무",
                        placeholder="예: 데이터 분석가 (광고·마케팅 도메인), LLM 에이전트 엔지니어",
                        lines=2,
                        elem_classes=["job-input"],
                    )
                    job_eval_btn = gr.Button("적합성 평가 실행", variant="primary", size="lg", elem_classes=["cta-button"])
                    with gr.Accordion("🔍 사용 중인 평가 프롬프트 (시스템 지시문)", open=False):
                        gr.Markdown(
                            f"```text\n{get_job_fit_prompt_placeholder_display()}\n```",
                            elem_classes=["prompt-preview"],
                        )
                    job_eval_out = gr.Markdown(elem_classes=["insight-output"])

                    def _run_job_eval(title: str):
                        if not os.getenv("OPENAI_API_KEY", "").strip():
                            return "⚠️ OPENAI_API_KEY가 필요합니다."
                        return evaluate_job_fit_for_role(title or "")

                    job_eval_btn.click(
                        fn=_run_job_eval,
                        inputs=[job_title_in],
                        outputs=[job_eval_out],
                    )

        def _check_pwd(pwd):
            expect = _app_password if _app_password else "dev"
            ok = (pwd or "").strip() == expect
            if ok:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    gr.update(visible=False),
                )
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                gr.update(value="⚠️ 비밀번호가 올바르지 않습니다.", visible=True),
            )

        pwd_btn.click(
            fn=_check_pwd,
            inputs=[pwd_input],
            outputs=[pwd_section, chat_section, pwd_input, pwd_error],
        )
    return demo


if __name__ == "__main__":
    custom_css = """
    .stats-bar { padding: 0.6rem 1rem; border-radius: 10px; background: var(--block-background-fill); border: 1px solid var(--block-border-color); }
    .chat-container { min-height: 420px; border-radius: 12px !important; }
    .app-hero {
      padding: 1.1rem 1.35rem;
      border-radius: 16px;
      margin-bottom: 0.75rem;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.14), rgba(14, 165, 233, 0.08));
      border: 1px solid rgba(99, 102, 241, 0.28);
      box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    }
    .app-hero-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; color: var(--body-text-color); margin-bottom: 0.25rem; }
    .app-hero-sub { font-size: 0.92rem; opacity: 0.88; line-height: 1.45; }
    .main-tabs { margin-top: 0.25rem; }
    .main-tabs .tab-nav button { font-weight: 600 !important; border-radius: 10px 10px 0 0 !important; }
    .panel-card {
      padding: 1rem 1.15rem;
      border-radius: 14px;
      border: 1px solid var(--block-border-color);
      background: var(--block-background-fill);
      margin-bottom: 1rem;
      box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
    }
    .panel-card-accent {
      border-color: rgba(99, 102, 241, 0.35);
      background: linear-gradient(180deg, rgba(99, 102, 241, 0.06), var(--block-background-fill));
    }
    .panel-heading { margin: 0 0 0.5rem 0; font-size: 1.08rem; font-weight: 650; }
    .panel-desc { margin: 0; font-size: 0.9rem; line-height: 1.55; opacity: 0.92; }
    .cta-button { min-height: 2.75rem !important; font-weight: 600 !important; border-radius: 10px !important; }
    .job-input textarea, .job-input input { border-radius: 10px !important; }
    .insight-output {
      padding: 1rem 1.15rem;
      border-radius: 12px;
      border: 1px solid var(--block-border-color);
      background: var(--block-background-fill);
      min-height: 120px;
    }
    .prompt-preview { font-size: 0.82rem !important; max-height: 320px; overflow: auto; }
    .gradio-container .block.examples, .gradio-container [class*="examples"] { display: none !important; }
    """
    theme = gr.themes.Glass(primary_hue="indigo", secondary_hue="slate")
    demo = build_ui()
    demo.launch(theme=theme, css=custom_css)
