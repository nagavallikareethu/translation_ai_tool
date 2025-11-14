"""
Enhanced MCQ generation pipeline.
Supports strict formatting for Indic languages, topic-only mode,
and Playwright-based PDF rendering with ReportLab fallback.
"""

import asyncio
import html
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from playwright.async_api import async_playwright
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# ======================================================
# LOAD GEMINI API KEY
# ======================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY or GENAI_API_KEY not found in .env file!")

genai.configure(api_key=api_key)
print("✅ Gemini API Key loaded successfully!\n")

SUPPORTED_LANGUAGES = ["English", "Telugu", "Hindi", "Odia"]
STRICT_LANGUAGES = {"hindi", "odia"}
LANG_DISPLAY = {
    "english": "English",
    "telugu": "Telugu",
    "hindi": "Hindi",
    "odia": "Odia",
}
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).parent.resolve()
FONTS_DIR = SCRIPT_DIR / "fonts"


# ======================================================
# Helpers
# ======================================================
def ensure_supported_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized not in LANG_DISPLAY:
        print(f"⚠️ Unsupported language '{language}'. Falling back to English.")
        return "english"
    return normalized


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def normalize_unicode_digits(text: str | None) -> str:
    if not text:
        return ""
    chars = []
    for ch in text:
        if ch.isdigit() and not ch.isascii():
            try:
                chars.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        chars.append(ch)
    return "".join(chars)


def clean_text_html(value: str | None) -> str:
    if not value:
        return ""
    cleaned = html.escape(str(value))
    return cleaned.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()


# ======================================================
# Prompt builders
# ======================================================
def _strict_example_block(language: str) -> str:
    if language == "hindi":
        return """1. दो और दो का योग क्या है?
A) 3
B) 4
C) 5
D) 6
Answer: B

2. भारत की राजधानी क्या है?
A) मुंबई
B) दिल्ली
C) कोलकाता
D) चेन्नई
Answer: B
"""
    if language == "odia":
        return """1. ଦୁଇ ଏବଂ ଦୁଇର ଯୋଗଫଳ କ'ଣ?
A) 3
B) 4
C) 5
D) 6
Answer: B

2. ଭାରତର ରାଜଧାନୀ କ'ଣ?
A) ମୁମ୍ବାଇ
B) ନୟା ଦିଲ୍ଲୀ
C) କୋଲକାତା
D) ଚେନ୍ନାଇ
Answer: B
"""
    return ""


def build_prompt(source_text: str,
                 num_questions: int,
                 target_language: str,
                 topic: Optional[str],
                 input_mode: str) -> str:
    language_display = LANG_DISPLAY[target_language]
    topic_clause = (
        f"- Focus strictly on the topic: \"{topic}\"."
        if topic else
        "- Use the most relevant parts of the source material."
    )
    strict_block = ""
    if target_language in STRICT_LANGUAGES:
        strict_block = f"""
CRITICAL {language_display.upper()} FORMATTING RULES:
- Write everything in {language_display}, but keep option markers and the word "Answer" in English.
- Options must use only A), B), C), D).
- Numbers must remain digits (20, ₹120); do not spell them out.

FOLLOW THIS EXAMPLE:
{_strict_example_block(target_language)}
"""

    return f"""
You are an expert exam-question generator.

Document mode: {input_mode.upper()}
{topic_clause}

Generate **exactly {num_questions}** multiple-choice questions in {language_display}.

Non-negotiable format:
1. Question line: "<number>. <question?>"
2. Option lines: "A) ...", "B) ...", "C) ...", "D) ..."
3. Answer line: "Answer: X"

{strict_block}

Source content (truncated):
{source_text[:10000]}

Begin now."""


# ======================================================
# Gemini interaction + post processing
# ======================================================
def debug_raw_output(text: str, language: str):
    print(f"\n=== RAW GEMINI OUTPUT ({language}) - first 1000 chars ===")
    print((text or "")[:1000])
    print("=== END RAW OUTPUT ===\n")


def fix_missing_option_markers(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    corrected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(")") and len(stripped) > 1:
            option_count = 0
            for prev in reversed(corrected):
                prev_line = prev.strip()
                if re.match(r"^\d+\.", prev_line):
                    break
                if re.match(r"^[A-D]\)", prev_line):
                    option_count += 1
            if option_count < 4:
                next_letter = chr(65 + option_count)
                corrected.append(f"{next_letter}{stripped}")
                continue
        corrected.append(line)
    return "\n".join(corrected)


def fix_missing_answers(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    corrected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {"answer:", "answer", "उत्तर:", "ଉତ୍ତର:"}:
            corrected.append("Answer: A")
        else:
            corrected.append(line)
    return "\n".join(corrected)


def correct_output(text: str, language: str) -> str:
    if not text:
        return ""
    text = fix_missing_option_markers(text)
    text = fix_missing_answers(text)
    text = re.sub(r"(\d+)\.(?!\s)", r"\1. ", text)
    text = re.sub(r"([A-D])\)(?!\s)", r"\1) ", text)
    return text


def generate_mcq_text(num_questions: int,
                      language: str,
                      source_text: str,
                      topic: Optional[str],
                      mode: str) -> str:
    prompt = build_prompt(source_text, num_questions, language, topic, mode)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    text = response.text or ""
    debug_raw_output(text, language)
    if language in STRICT_LANGUAGES:
        text = correct_output(text, language)
    return text


# ======================================================
# PDF Rendering
# ======================================================
def parse_mcq_text(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    questions = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        q_match = re.match(r"^(\d+)\.\s*(.+\?)", stripped)
        if q_match:
            if current:
                questions.append(current)
            current = {
                "number": q_match.group(1),
                "content": q_match.group(2),
                "options": [],
                "answer": "",
            }
            continue
        if current and re.match(r"^[A-D]\)", stripped):
            current["options"].append(stripped)
            continue
        if current and stripped.lower().startswith("answer"):
            current["answer"] = stripped if stripped.startswith("Answer:") else f"Answer: {stripped.split(':')[-1].strip()}"
            continue
    if current:
        questions.append(current)
    return questions


async def render_pdf_playwright(html_content: str, output_path: Path):
    tmpdir = tempfile.mkdtemp()
    html_path = Path(tmpdir) / "mcqs.html"
    html_path.write_text(html_content, encoding="utf-8")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html_path.resolve().as_uri())
        await page.pdf(
            path=str(output_path),
            format="A4",
            margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"},
            print_background=True,
        )
        await browser.close()


def build_html_document(questions: List[Dict[str, str]], language: str, raw_text: str) -> str:
    font_map = {
        "english": None,
        "telugu": "NotoSansTelugu-Regular.ttf",
        "hindi": "TiroDevanagariHindi-Regular.ttf",
        "odia": "NotoSansOriya-Regular.ttf",
    }
    font_file = font_map.get(language)
    font_face = ""
    body_font = "Arial, sans-serif"
    if font_file:
        abs_path = (FONTS_DIR / font_file)
        if abs_path.exists():
            font_face = f"""
            @font-face {{
                font-family: 'LangFont';
                src: url('{abs_path.resolve().as_uri()}') format('truetype');
            }}
            """
            body_font = "LangFont, Arial, sans-serif"

    css = f"""
    {font_face}
    body {{
        font-family: {body_font};
        padding: 32px;
        line-height: 1.6;
    }}
    .question {{
        border-bottom: 1px solid #ddd;
        margin-bottom: 24px;
        padding-bottom: 16px;
    }}
    .question-number {{
        font-weight: bold;
    }}
    .answer {{
        color: #0a730a;
        font-weight: bold;
        margin-top: 8px;
    }}
    """

    if questions:
        content = []
        for q in questions:
            block = ["<div class='question'>"]
            block.append(f"<div class='question-number'>Question {clean_text_html(q.get('number'))}</div>")
            block.append(f"<div>{clean_text_html(q.get('content'))}</div>")
            if q.get("options"):
                block.append("<ul>")
                for opt in q["options"]:
                    block.append(f"<li>{clean_text_html(opt)}</li>")
                block.append("</ul>")
            if q.get("answer"):
                block.append(f"<div class='answer'>{clean_text_html(q['answer'])}</div>")
            block.append("</div>")
            content.append("\n".join(block))
        body = "\n".join(content)
    else:
        body = f"<pre>{clean_text_html(raw_text)}</pre>"

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <style>{css}</style>
    </head>
    <body>
        <h1>Generated MCQs - {LANG_DISPLAY.get(language, language.title())}</h1>
        {body}
    </body>
    </html>
    """


def save_pdf(text: str, outpath: Path, lang: str) -> bool:
    questions = parse_mcq_text(text)
    html_content = build_html_document(questions, lang, text)
    try:
        asyncio.run(render_pdf_playwright(html_content, outpath))
        return True
    except Exception as exc:
        print(f"Playwright rendering failed: {exc}, falling back to ReportLab.")

    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    font_map = {
        "english": ("Helvetica", None),
        "telugu": ("NotoSansTelugu", "fonts/NotoSansTelugu-Regular.ttf"),
        "hindi": ("TiroDevanagariHindi", "fonts/TiroDevanagariHindi-Regular.ttf"),
        "odia": ("NotoSansOriya", "fonts/NotoSansOriya-Regular.ttf"),
    }
    font_name, font_file = font_map.get(lang, ("Helvetica", None))
    if font_file and (FONTS_DIR / Path(font_file).name).exists():
        pdfmetrics.registerFont(TTFont(font_name, str(FONTS_DIR / Path(font_file).name)))
    else:
        font_name = "Helvetica"

    c = canvas.Canvas(str(outpath), pagesize=A4)
    c.setFont(font_name, 12)
    width, height = A4
    y = height - 60
    for line in clean_text.splitlines():
        if y < 60:
            c.showPage()
            c.setFont(font_name, 12)
            y = height - 60
        c.drawString(40, y, line[:150])
        y -= 18
    c.save()
    return True


# ======================================================
# Public API
# ======================================================
def generate_mcqs_content(pdf_path: Optional[str],
                          n: int,
                          language: str,
                          topic: Optional[str] = None,
                          custom_context: Optional[str] = None) -> str:
    language_code = ensure_supported_language(language)
    if pdf_path:
        source_text = extract_text_from_pdf(pdf_path)
        if not source_text:
            raise ValueError("⚠️ No readable text found in the PDF! Make sure it’s not just scanned images.")
        mode = "pdf"
    else:
        source_text = (custom_context or topic or "").strip()
        if not source_text:
            raise ValueError("⚠️ Provide either a PDF or dynamic text to generate MCQs.")
        mode = "text"
    return generate_mcq_text(n, language_code, source_text, topic, mode)


def run_mcq_pipeline(pdf_path: Optional[str],
                     num_questions: int,
                     language: str,
                     topic: Optional[str] = None,
                     custom_context: Optional[str] = None,
                     output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Tuple[str, Optional[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcq_text = generate_mcqs_content(pdf_path, num_questions, language, topic, custom_context)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(pdf_path).stem[:30].replace(" ", "_") if pdf_path else "custom_text"
    pdf_name = f"mcqs_{safe_name}_{language.lower()}_{timestamp}.pdf"
    out_path = output_dir / pdf_name
    pdf_ok = save_pdf(mcq_text, out_path, ensure_supported_language(language))
    return mcq_text, str(out_path if pdf_ok else "")


# ======================================================
# CLI (optional)
# ======================================================
def cli_main():
    pdf_path = input("📂 Enter PDF path (leave blank for topic mode): ").strip()
    if pdf_path and not Path(pdf_path).exists():
        raise FileNotFoundError("⚠️ File not found! Please enter a valid PDF path.")

    num_qs = int(input("🧮 How many MCQs to generate?: ").strip())

    print("\n🌐 Available Languages:")
    for i, lang_option in enumerate(SUPPORTED_LANGUAGES, 1):
        print(f"{i}. {lang_option}")
    choice = int(input("\n👉 Enter the number of your language choice: ").strip())
    if choice < 1 or choice > len(SUPPORTED_LANGUAGES):
        raise ValueError("Invalid choice! Please select a valid option.")
    lang = SUPPORTED_LANGUAGES[choice - 1]

    topic = input("🎯 Optional topic focus (press Enter to skip): ").strip() or None
    dynamic_text = input("📝 Optional custom passage (press Enter to skip): ").strip() or None

    print("\n🧠 Generating MCQs using Gemini 2.5 Pro... please wait\n")
    mcqs, pdf_path = run_mcq_pipeline(pdf_path or None, num_qs, lang, topic, dynamic_text)
    if mcqs:
        print(f"\n✅ Generation complete. PDF: {pdf_path}")
    else:
        print("\n⚠️ No MCQs generated.")


if __name__ == "__main__":
    cli_main()