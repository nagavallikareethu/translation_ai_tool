import os
import re
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

# ======================================================
# FILE INPUT (same as solving)
# ======================================================
pdf_path = input("📂 Enter your PDF file path: ").strip()

if not os.path.exists(pdf_path):
    raise FileNotFoundError("⚠️ File not found! Please enter a valid PDF path.")

num_qs = int(input("🧮 How many MCQs to generate?: ").strip())

languages = ["English", "Telugu", "Hindi", "Odia"]
print("\n🌐 Available Languages:")
for i, lang in enumerate(languages, 1):
    print(f"{i}. {lang}")

choice = int(input("\n👉 Enter the number of your language choice: ").strip())
if choice < 1 or choice > len(languages):
    raise ValueError("Invalid choice! Please select a valid option.")

lang = languages[choice - 1]

# ======================================================
# PDF TEXT EXTRACTION FUNCTION
# ======================================================
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ======================================================
# GEMINI CALL FUNCTION
# ======================================================
def generate_mcqs(pdf_path, n, language):
    pdf_text = extract_text_from_pdf(pdf_path)

    if not pdf_text:
        raise ValueError("⚠️ No readable text found in the PDF! Make sure it’s not just scanned images.")

    prompt = f"""
    You are an expert exam question generator.

    Read the following document carefully and generate {n} *new* MCQs.

    Rules:
    - Write all questions, options, and answers completely in {language} language only.
    - Do NOT include any English translations or mixed language.
    - Each question must have 4 options (A, B, C, D).
    - Clearly mention the correct answer after each question as:
      Answer: <Option>
    - Keep questions concept-based, short, and professional.

    Document content:
    {pdf_text[:10000]}
    """

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

# ======================================================
# SAVE TO PDF FUNCTION
# ======================================================
def save_pdf(text, outpath, lang):
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

    c = canvas.Canvas(outpath, pagesize=A4)
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
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    print("\n🧠 Generating MCQs using Gemini 2.5 Pro... please wait\n")
    mcqs = generate_mcqs(pdf_path, num_qs, lang)

    if mcqs:
        output_pdf = f"Generated_MCQs_{lang}.pdf"
        ok = save_pdf(mcqs, output_pdf, lang)

        if ok:
            print(f"\n✅ {lang} PDF generated successfully: {output_pdf}")
        else:
            print("\n❌ Failed to create PDF.")
    else:
        print("\n⚠️ No MCQs generated.")
