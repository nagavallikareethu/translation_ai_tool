import os
import io
import tempfile
import pathlib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Backends
import importlib

# Ensure project working directory
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)

# Load env and normalize Gemini keys between scripts
load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GENAI_API_KEY"):
    os.environ["GENAI_API_KEY"] = os.getenv("GEMINI_API_KEY")
if os.getenv("GENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GENAI_API_KEY")


# Lazy imports so the app can boot even if optional deps are missing
def _lazy_import_translation():
    tr = importlib.import_module("translate")
    return tr.PDFToJSONConverter, tr.JSONTranslator, tr.PDFGenerator


def _lazy_import_solution():
    sol = importlib.import_module("solution")
    return (
        sol.extract_pdf,
        sol.solve_pages,
        sol.translate_items,
        sol.render_pdf_from_data,
    )


def _lazy_import_generate():
    gen = importlib.import_module("generate")
    return gen.extract_text_from_pdf, gen.generate_mcqs, gen.save_pdf


# Shared helpers
def ensure_session_dirs():
    if "work_dir" not in st.session_state:
        base = pathlib.Path(tempfile.mkdtemp(prefix="st_pdf_"))
        st.session_state.work_dir = str(base)
        (base / "outputs").mkdir(exist_ok=True)
    return pathlib.Path(st.session_state.work_dir)


def write_uploaded_pdf(uploaded_file) -> pathlib.Path:
    work_dir = ensure_session_dirs()
    pdf_path = work_dir / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["uploaded_pdf_path"] = str(pdf_path)
    return pdf_path


def file_bytes_for_download(path: pathlib.Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# UI
st.set_page_config(page_title="AI PDF Workspace", page_icon="🧠", layout="wide")
st.title("🧠 Unified AI PDF Workspace")
st.caption("Translate • Solve • Generate MCQs — using a single uploaded PDF")

with st.sidebar:
    st.header("1) Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None:
        pdf_path = write_uploaded_pdf(uploaded)
        st.success(f"Uploaded: {uploaded.name} ({uploaded.size/1024:.1f} KB)")
        st.caption(f"Saved to: {pdf_path}")
    else:
        pdf_path = None


tabs = st.tabs(["Translation", "Solution", "MCQ Generation"])  # 3 modules


# -------------------------
# Translation Module Tab
# -------------------------
with tabs[0]:
    st.subheader("🌐 Translation Module")
    st.caption("Translate the PDF content into the selected language")

    # Limit to requested languages
    lang_display_to_code = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi",
        "Odia": "or",
        "Tamil": "ta",
    }

    t_lang_name = st.selectbox("Target language", list(lang_display_to_code.keys()), index=1)
    translate_btn = st.button("Translate", type="primary", disabled=pdf_path is None)

    if translate_btn:
        if not pdf_path:
            st.error("Please upload a PDF first.")
        else:
            try:
                PDFToJSONConverter, JSONTranslator, PDFGenerator = _lazy_import_translation()
                work_dir = ensure_session_dirs()
                extracted_json = work_dir / "extracted_pdf.json"
                output_pdf = work_dir / f"translated_{t_lang_name.lower()}.pdf"

                with st.spinner("Extracting PDF → JSON ..."):
                    converter = PDFToJSONConverter()
                    data = converter.convert_pdf_to_json_enhanced(
                        str(pdf_path), output_path=str(extracted_json), include_images=True, image_handling="metadata"
                    )

                with st.spinner(f"Translating content → {t_lang_name} ..."):
                    translator = JSONTranslator()
                    translated_json_path = translator.translate_json_file(
                        str(extracted_json), lang_display_to_code[t_lang_name], t_lang_name
                    )

                with st.spinner("Rendering translated PDF ..."):
                    pdf_gen = PDFGenerator(str(translated_json_path), str(output_pdf))
                    pdf_gen.generate_pdf()

                st.success("Translation complete!")
                st.session_state["translated_pdf_path"] = str(output_pdf)

            except Exception as e:
                st.error(f"Translation failed: {e}")

    if st.session_state.get("translated_pdf_path"):
        out_path = pathlib.Path(st.session_state["translated_pdf_path"])
        st.download_button(
            label="Download Translated PDF",
            data=file_bytes_for_download(out_path),
            file_name=out_path.name,
            mime="application/pdf",
        )


# -------------------------
# Solution Module Tab
# -------------------------
with tabs[1]:
    st.subheader("🧩 Solution Module")
    st.caption("Extract, solve, translate, and render solutions")

    sol_langs = ["Telugu", "Hindi", "Odia", "Tamil", "Kannada", "English"]
    s_lang_name = st.selectbox("Solution output language", sol_langs, index=0, key="sol_lang")
    solve_btn = st.button("Solve", type="primary", disabled=pdf_path is None, key="solve_btn")

    if solve_btn:
        if not pdf_path:
            st.error("Please upload a PDF first.")
        else:
            try:
                extract_pdf, solve_pages, translate_items, render_pdf_from_data = _lazy_import_solution()
                work_dir = ensure_session_dirs()
                out_pdf = work_dir / f"final_solved_{s_lang_name.lower()}.pdf"

                with st.spinner("Extracting PDF (text + images) ..."):
                    pages = extract_pdf(str(pdf_path), output_json=str(work_dir / "extracted_data.json"), output_image_folder=str(work_dir / "extracted_images"))

                with st.spinner("Solving extracted content ..."):
                    solved = solve_pages(pages)

                with st.spinner(f"Translating solutions → {s_lang_name} ..."):
                    translated = translate_items(solved, s_lang_name)

                with st.spinner("Rendering solved PDF ..."):
                    # solution.render_pdf_from_data is async wrapper-friendly
                    import asyncio
                    asyncio.run(render_pdf_from_data(translated, s_lang_name.lower(), str(out_pdf)))

                st.success("Solved PDF is ready!")
                st.session_state["solved_pdf_path"] = str(out_pdf)

            except Exception as e:
                st.error(f"Solving failed: {e}")

    if st.session_state.get("solved_pdf_path"):
        out_path = pathlib.Path(st.session_state["solved_pdf_path"])
        st.download_button(
            label="Download Solved PDF",
            data=file_bytes_for_download(out_path),
            file_name=out_path.name,
            mime="application/pdf",
        )


# -------------------------
# MCQ Generation Module Tab
# -------------------------
with tabs[2]:
    st.subheader("❓ MCQ Generation Module")
    st.caption("Generate new MCQs from the uploaded PDF")

    mcq_langs = ["English", "Telugu", "Hindi", "Odia"]
    m_lang = st.selectbox("Target language", mcq_langs, index=0, key="mcq_lang")
    m_count = st.number_input("Number of MCQs", min_value=1, max_value=100, value=10, step=1)
    gen_btn = st.button("Generate MCQs", type="primary", disabled=pdf_path is None)

    if gen_btn:
        if not pdf_path:
            st.error("Please upload a PDF first.")
        else:
            try:
                extract_text_from_pdf, generate_mcqs, save_pdf = _lazy_import_generate()
                work_dir = ensure_session_dirs()
                out_pdf = work_dir / f"Generated_MCQs_{m_lang}.pdf"

                with st.spinner("Generating MCQs using Gemini ..."):
                    text = generate_mcqs(str(pdf_path), int(m_count), m_lang)

                if not text:
                    st.warning("No MCQs were generated.")
                else:
                    st.text_area("MCQs", value=text, height=300)
                    with st.spinner("Saving MCQs to PDF ..."):
                        ok = save_pdf(text, str(out_pdf), m_lang)
                    if ok:
                        st.success("MCQ PDF is ready!")
                        st.session_state["mcq_pdf_path"] = str(out_pdf)
                    else:
                        st.error("Failed to create MCQ PDF.")

            except Exception as e:
                st.error(f"MCQ generation failed: {e}")

    if st.session_state.get("mcq_pdf_path"):
        out_path = pathlib.Path(st.session_state["mcq_pdf_path"])
        st.download_button(
            label="Download MCQ PDF",
            data=file_bytes_for_download(out_path),
            file_name=out_path.name,
            mime="application/pdf",
        )


