import os
import shutil
import textwrap
import time
from pathlib import Path
from typing import Tuple

import gradio as gr

from translation import (
    run_full_pipeline,
    LANG_CODE_TO_NAME,
    LANG_NAME_TO_CODE,
)
from solution import run_solution_pipeline, LANGUAGES as SOLUTION_LANGUAGES
from generation import run_mcq_pipeline, SUPPORTED_LANGUAGES as MCQ_LANGUAGES

BASE_UI_DIR = Path("ui_workspace")
UPLOAD_DIR = BASE_UI_DIR / "uploads"
OUTPUT_DIR = BASE_UI_DIR / "outputs"
TRANSLATION_DIR = OUTPUT_DIR / "translations"
SOLUTION_DIR = OUTPUT_DIR / "solutions"
MCQ_DIR = OUTPUT_DIR / "mcqs"

for folder in [UPLOAD_DIR, OUTPUT_DIR, TRANSLATION_DIR, SOLUTION_DIR, MCQ_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TRANSLATION_LANGUAGE_OPTIONS = list(LANG_CODE_TO_NAME.values())
SOLUTION_LANGUAGE_OPTIONS = sorted(set(SOLUTION_LANGUAGES.values()))
MCQ_LANGUAGE_OPTIONS = MCQ_LANGUAGES
TOPIC_OPTIONS = [
    "Quantitative Aptitude",
    "Reasoning Ability",
    "English Language",
    "General Awareness",
    "Computer Knowledge",
    "Data Interpretation",
]


def _copy_uploaded_file(file) -> Tuple[str | None, str]:
    if not file:
        return None, "⚠️ Please upload a PDF to get started."

    source_path = getattr(file, "name", None) or file
    original_name = getattr(file, "orig_name", None) or os.path.basename(source_path)
    if not os.path.exists(source_path):
        return None, "⚠️ Unable to read the uploaded file. Please try again."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target_path = UPLOAD_DIR / f"{timestamp}_{original_name}"
    shutil.copy(source_path, target_path)

    file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    info = textwrap.dedent(
        f"""
        ✅ **File received**
        - Name: `{original_name}`
        - Size: {file_size_mb:.2f} MB
        - Stored at: `{target_path}`
        """
    ).strip()
    return str(target_path), info


def _ensure_pdf(pdf_path: str | None) -> Tuple[bool, str]:
    if not pdf_path or not Path(pdf_path).exists():
        return False, "⚠️ Upload a PDF before running this module."
    return True, ""


def handle_file_upload(file):
    return _copy_uploaded_file(file)


def trigger_translation(pdf_path: str | None, language_name: str):
    ok, message = _ensure_pdf(pdf_path)
    if not ok:
        return message, None

    lang_code = LANG_NAME_TO_CODE.get(language_name.lower())
    if not lang_code:
        return f"⚠️ Unsupported language: {language_name}", None

    translation_output_dir = TRANSLATION_DIR / lang_code
    translation_json_dir = translation_output_dir / "json"
    translation_output_dir.mkdir(parents=True, exist_ok=True)
    translation_json_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_full_pipeline(
            pdf_path=pdf_path,
            languages=[lang_code],
            include_images=True,
            image_handling="metadata",
            translated_dir=str(translation_json_dir),
            output_dir=str(translation_output_dir),
            overlay=True,
        )
        generated_pdfs = result.get("generated_pdfs") or []
        if not generated_pdfs:
            return "⚠️ Translation finished but no PDF was produced.", None

        pdf_file = generated_pdfs[0]
        summary = textwrap.dedent(
            f"""
            ✅ Translation complete!
            - Language: **{language_name}**
            - PDF: `{pdf_file}`
            """
        ).strip()
        return summary, pdf_file
    except Exception as exc:
        return f"❌ Translation failed:\n```\n{exc}\n```", None


def trigger_solution(pdf_path: str | None, language_name: str):
    ok, message = _ensure_pdf(pdf_path)
    if not ok:
        return message, None

    try:
        result = run_solution_pipeline(
            pdf_path=pdf_path,
            target_language=language_name,
            output_dir=str(SOLUTION_DIR),
        )
        pdf_file = result.get("output_pdf")
        summary = textwrap.dedent(
            f"""
            ✅ Solution generation complete!
            - Language: **{language_name}**
            - Solved questions: {len(result.get("solved_items", []))}
            - PDF: `{pdf_file}`
            """
        ).strip()
        return summary, pdf_file
    except Exception as exc:
        return f"❌ Solution generation failed:\n```\n{exc}\n```", None


def trigger_mcq_generation(pdf_path: str | None,
                           mode: str,
                           num_questions: int,
                           language_name: str,
                           topic_choice: str,
                           custom_topic: str,
                           dynamic_text: str):
    if mode == "pdf":
        ok, message = _ensure_pdf(pdf_path)
        if not ok:
            return message, "", None
        source_pdf = pdf_path
    else:
        source_pdf = None
        final_topic = (custom_topic.strip() or topic_choice or "").strip()
        if not (dynamic_text.strip() or final_topic):
            return "⚠️ Provide a topic or some custom text when using text-only mode.", "", None

    final_topic = custom_topic.strip() or topic_choice
    context_payload = dynamic_text.strip() or final_topic
    try:
        mcq_text, pdf_file = run_mcq_pipeline(
            pdf_path=source_pdf,
            num_questions=num_questions,
            language=language_name,
            topic=final_topic,
            custom_context=context_payload,
            output_dir=str(MCQ_DIR),
        )
        preview = mcq_text.replace("Answer:", "**Answer:**")
        summary = textwrap.dedent(
            f"""
            ✅ Generated {num_questions} MCQs
            - Language: **{language_name}**
            - Mode: **{"PDF" if source_pdf else "Text-only"}**
            - Topic: **{final_topic or 'Derived from input'}**
            - PDF: `{pdf_file}`
            """
        ).strip()
        return summary, preview, pdf_file if pdf_file else None
    except Exception as exc:
        return f"❌ MCQ generation failed:\n```\n{exc}\n```", "", None


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📘 Unified AI Exam Assistant
        Upload a single PDF and run **Translation**, **Solution Generation**, or **MCQ Generation**
        using the existing AI pipelines.
        """
    )

    pdf_state = gr.State()

    with gr.Row():
        pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"], file_count="single")
        file_info = gr.Markdown("⬆️ Upload a PDF to get started.")

    pdf_input.upload(handle_file_upload, pdf_input, outputs=[pdf_state, file_info])

    with gr.Tab("Translation"):
        translation_language = gr.Dropdown(
            TRANSLATION_LANGUAGE_OPTIONS,
            value=TRANSLATION_LANGUAGE_OPTIONS[0],
            label="Target Language",
        )
        translate_button = gr.Button("Translate PDF", variant="primary")
        translation_status = gr.Markdown()
        translation_download = gr.File(label="Download translated PDF")

        translate_button.click(
            trigger_translation,
            inputs=[pdf_state, translation_language],
            outputs=[translation_status, translation_download],
        )

    with gr.Tab("Solution Generator"):
        solution_language = gr.Dropdown(
            SOLUTION_LANGUAGE_OPTIONS,
            value=SOLUTION_LANGUAGE_OPTIONS[0],
            label="Solution Output Language",
        )
        solve_button = gr.Button("Solve & Translate", variant="primary")
        solution_status = gr.Markdown()
        solution_download = gr.File(label="Download solved PDF")

        solve_button.click(
            trigger_solution,
            inputs=[pdf_state, solution_language],
            outputs=[solution_status, solution_download],
        )

    with gr.Tab("MCQ Generation"):
        mcq_mode = gr.Radio(["pdf", "text"], value="pdf", label="Input Mode", info="Use uploaded PDF or custom text?")
        mcq_count = gr.Slider(5, 50, value=10, step=1, label="Number of MCQs")
        mcq_language = gr.Dropdown(
            MCQ_LANGUAGE_OPTIONS,
            value=MCQ_LANGUAGE_OPTIONS[0],
            label="MCQ Language",
        )
        topic_dropdown = gr.Dropdown(
            TOPIC_OPTIONS,
            value=TOPIC_OPTIONS[0],
            label="Select Topic",
        )
        custom_topic = gr.Textbox(
            label="Custom Topic (optional)",
            placeholder="Enter a custom topic if needed",
        )
        dynamic_text = gr.Textbox(
            label="Dynamic Context / Custom Passage (optional)",
            placeholder="Paste any text you want the MCQs to consider",
            lines=6,
        )
        mcq_button = gr.Button("Generate MCQs", variant="primary")
        mcq_status = gr.Markdown()
        mcq_preview = gr.Markdown(label="Preview")
        mcq_download = gr.File(label="Download MCQ PDF")

        mcq_button.click(
            trigger_mcq_generation,
            inputs=[pdf_state, mcq_mode, mcq_count, mcq_language, topic_dropdown, custom_topic, dynamic_text],
            outputs=[mcq_status, mcq_preview, mcq_download],
        )

demo.queue()


if __name__ == "__main__":
    demo.launch()

