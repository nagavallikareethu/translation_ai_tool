import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ======================================================
# LOAD GEMINI API KEY
# ======================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

genai.configure(api_key=api_key)
print("✅ Gemini API Key loaded successfully!\n")

SUPPORTED_LANGUAGES = ["English", "Telugu", "Hindi", "Odia"]
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# PDF TEXT EXTRACTION FUNCTION
# ======================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# ======================================================
# GEMINI PROMPT + CALL FUNCTIONS
# ======================================================
def _build_mcq_prompt(source_text: str,
                      n: int,
                      language: str,
                      topic: Optional[str] = None,
                      custom_context: Optional[str] = None,
                      input_mode: str = "pdf") -> str:
    topic_clause = (
        f"The MCQs must stay tightly aligned to the topic: **{topic}**."
        if topic else
        "Derive the most relevant subtopics directly from the provided material."
    )
    extra_context = custom_context.strip() if custom_context else ""
    supplemental = (
        f"\nSupplementary context:\n{extra_context}\n"
        if extra_context else
        "\nNo supplemental context was provided."
    )
    source_label = "PDF extract" if input_mode == "pdf" else "User-provided text"

    return f"""
You are an expert exam-question generator.

Task:
- Read the provided {source_label} and craft **{n} brand new MCQs** in {language}.
- {topic_clause}

Hard Requirements:
- Author concise, professional questions suitable for competitive exams.
- Every question must include four options labeled A, B, C, D.
- After the options, explicitly state the answer exactly as `Answer: <Option Letter>`.
- Use **only** the {language} language (no transliteration, no mixing).
- Avoid reusing sentences verbatim from the source; paraphrase when needed.
- Blend context from both the main source and the supplemental text (if any).

Primary content (truncated to 10k characters):
{source_text[:10000]}

{supplemental}
""".strip()


def generate_mcqs_content(pdf_path: Optional[str],
                          n: int,
                          language: str,
                          topic: Optional[str] = None,
                          custom_context: Optional[str] = None) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported options: {', '.join(SUPPORTED_LANGUAGES)}")

    if pdf_path:
        source_text = extract_text_from_pdf(pdf_path)
        if not source_text:
            raise ValueError("⚠️ No readable text found in the PDF! Make sure it’s not just scanned images.")
        mode = "pdf"
    else:
        source_text = (custom_context or "").strip()
        if not source_text:
            raise ValueError("⚠️ Provide either a PDF or custom text to generate MCQs.")
        mode = "text"

    prompt = _build_mcq_prompt(source_text, n, language, topic, custom_context, input_mode=mode)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text


def generate_mcqs(pdf_path: str, n: int, language: str) -> str:
    return generate_mcqs_content(pdf_path, n, language)


# ======================================================
# SAVE TO PDF FUNCTION
# ======================================================
def save_pdf(text: str, outpath: str | Path, lang: str) -> bool:
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    clean_text = clean_text.replace("—", "-").replace("–", "-")

    # Language-wise font mapping
    font_map = {
        "English": ("Helvetica", None),
        "Telugu": ("NotoSansTelugu", "fonts/NotoSansTelugu-Regular.ttf"),
        "Hindi": ("NotoSansDevanagari", "fonts/NotoSansDevanagari-Regular.ttf"),
        "Odia": ("NotoSansOdia", "fonts/NotoSansOdia-Regular.ttf"),
    }

    font_name, font_file = font_map.get(lang, ("Helvetica", None))
    if font_file and os.path.exists(font_file):
        pdfmetrics.registerFont(TTFont(font_name, font_file))
    else:
        font_name = "Helvetica"

    c = canvas.Canvas(str(outpath), pagesize=A4)
    c.setFont(font_name, 13)
    width, height = A4
    y = height - 80
    max_chars_per_line = 85

    for line in clean_text.split("\n"):
        line = line.strip()
        if not line:
            y -= 15
            continue

        while len(line) > max_chars_per_line:
            part = line[:max_chars_per_line]
            c.drawString(60, y, part)
            y -= 20
            line = line[max_chars_per_line:]
            if y < 60:
                c.showPage()
                c.setFont(font_name, 13)
                y = height - 80

        if line:
            c.drawString(60, y, line)
            y -= 20

        if y < 60:
            c.showPage()
            c.setFont(font_name, 13)
            y = height - 80

    c.save()
    return os.path.exists(outpath)


# ======================================================
# PROGRAMMATIC PIPELINE
# ======================================================
def run_mcq_pipeline(pdf_path: Optional[str],
                     num_questions: int,
                     language: str,
                     topic: Optional[str] = None,
                     custom_context: Optional[str] = None,
                     output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Tuple[str, Optional[str]]:
    """
    Generates MCQs and writes them to a PDF, returning the raw text and PDF path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcq_text = generate_mcqs_content(pdf_path, num_questions, language, topic, custom_context)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(pdf_path).stem[:40].replace(" ", "_") if pdf_path else "custom_text"
    pdf_name = f"mcqs_{safe_name}_{language.lower()}_{timestamp}.pdf"
    pdf_path_out = output_dir / pdf_name

    pdf_ok = save_pdf(mcq_text, pdf_path_out, language)
    return mcq_text, str(pdf_path_out if pdf_ok else "")


# ======================================================
# CLI ENTRY POINT (preserves original behavior)
# ======================================================
def cli_main():
    pdf_path = input("📂 Enter your PDF file path: ").strip()
    if not os.path.exists(pdf_path):
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
    dynamic_text = input("📝 Optional supplemental context (press Enter to skip): ").strip() or None

    print("\n🧠 Generating MCQs using Gemini 2.5 Pro... please wait\n")
    mcqs, pdf_path_out = run_mcq_pipeline(pdf_path, num_qs, lang, topic, dynamic_text)

    if mcqs:
        if pdf_path_out:
            print(f"\n✅ {lang} PDF generated successfully: {pdf_path_out}")
        else:
            print("\n⚠️ MCQs generated but PDF creation failed.")
    else:
        print("\n⚠️ No MCQs generated.")


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    cli_main()