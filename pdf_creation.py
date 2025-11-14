import json
import os
import io
import re
import math
import html
import base64
import tempfile
from pathlib import Path
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from playwright.sync_api import sync_playwright

# -------------------- CONFIG --------------------
FONTS = {
    "telugu": r"E:\PDFExtraction\fonts\NotoSansTelugu-Regular.ttf",
    "hindi": r"E:\PDFExtraction\fonts\TiroDevanagariHindi-Regular.ttf",
    "odia": r"E:\PDFExtraction\fonts\AnekOdia-Regular.ttf"
}

# -------------------- UTILS --------------------
def clean(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(str(s))
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()

def detect_language(data) -> str:
    sample = json.dumps(data, ensure_ascii=False)
    if "telugu" in sample.lower():
        return "telugu"
    if "hindi" in sample.lower():
        return "hindi"
    if "odia" in sample.lower() or "oriya" in sample.lower():
        return "odia"
    # fallback to unicode-range heuristic
    hindi = re.findall(r'[\u0900-\u097F]', sample)
    telugu = re.findall(r'[\u0C00-\u0C7F]', sample)
    odia = re.findall(r'[\u0B00-\u0B7F]', sample)
    if len(hindi) >= len(telugu) and len(hindi) >= len(odia):
        return "hindi"
    if len(telugu) >= len(hindi) and len(telugu) >= len(odia):
        return "telugu"
    if len(odia) >= len(hindi) and len(odia) >= len(telugu):
        return "odia"
    return "telugu"

def _page_height_from(page_data):
    dims = page_data.get("dimensions")
    if isinstance(dims, dict):
        return dims.get("height", 842)
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        return dims[1]
    return 842

def _top_from_bottom(y_bottom, elem_height, page_height):
    return page_height - (y_bottom + elem_height)

# -------------------- HTML PAGE BUILDER (Playwright) --------------------
def build_page_html(page_data, lang):
    font_file = FONTS.get(lang, FONTS["telugu"])
    font_path = Path(font_file).resolve().as_uri() if os.path.exists(font_file) else ""

    css = f"""
    @font-face {{
        font-family: 'LangFont';
        src: url('{font_path}') format('truetype');
    }}
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:LangFont,sans-serif;width:595px;height:842px;background:#fff}}
    .text-element{{position:absolute;white-space:pre-wrap;background:transparent;color:#000;z-index:3}}
    .image-element{{position:absolute;z-index:2;border:none}}
    .line-element{{position:absolute;background:#000;transform-origin:0 0;z-index:3}}
    .rect-element{{position:absolute;border:1px solid #000;background:transparent;z-index:3}}
    """

    page_h = _page_height_from(page_data)
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<style>{css}</style></head><body>"
    ]

    # full white background (covers original visually)
    html_parts.append(f'<div style="position:absolute;left:0;top:0;width:595px;height:{page_h}px;background:#ffffff;z-index:0;"></div>')

    # images
    for image in page_data.get("images", []):
        pos = image.get("position", {})
        x0 = pos.get("x0", 0)
        width = pos.get("width", 0)
        height = pos.get("height", 0)
        y0 = _top_from_bottom(pos.get("y0", 0), height, page_h)
        img_b64 = image.get("data", "")
        img_fmt = image.get("image_format", "png").lower()
        if img_fmt == "jpg":
            img_fmt = "jpeg"
        if img_b64:
            try:
                img_src = f"data:image/{img_fmt};base64,{img_b64}"
                html_parts.append(f'<img class="image-element" src="{img_src}" style="left:{x0}px;top:{y0}px;width:{width}px;height:{height}px;">')
            except Exception:
                pass

    # text - ONLY ALLOWED CONTENT TYPES
    allowed_content_types = ["question", "exam_reference", "normal_text", "header", "instruction"]
    for tb in page_data.get("text_content", []):
        content_type = tb.get("content_type", "")
        if content_type not in allowed_content_types:
            continue
            
        content = tb.get("translated_content") or tb.get("content") or ""
        if not content:
            continue
        pos = tb.get("position", {})
        font_size = tb.get("font", {}).get("size", 12)
        x0 = pos.get("x0", 0)
        y0 = _top_from_bottom(pos.get("y0", 0), font_size, page_h)
        safe = clean(content)
        html_parts.append(f'<div class="text-element" style="left:{x0}px;top:{y0}px;font-size:{font_size}px;line-height:1.15;">{safe}</div>')

    # lines and rects
    for el in page_data.get("layout_elements", []):
        if el.get("type") == "line":
            x0 = el['position']['x0']; y0b = el['position']['y0']
            x1 = el['position']['x1']; y1b = el['position']['y1']
            y0 = _top_from_bottom(y0b, 0, page_h); y1 = _top_from_bottom(y1b, 0, page_h)
            width_len = math.hypot(x1 - x0, y1 - y0)
            angle = -math.atan2(y1 - y0, x1 - x0) if x1 != x0 else -math.pi/2
            html_parts.append(f'<div class="line-element" style="left:{x0}px;top:{y0}px;width:{width_len}px;height:1px;transform:rotate({angle}rad);"></div>')
    for sym in page_data.get("vector_symbols", []):
        x0, y0o, x1, y1o = sym['bbox']
        h = y1o - y0o
        y0 = _top_from_bottom(y0o, h, page_h)
        w = x1 - x0
        html_parts.append(f'<div class="rect-element" style="left:{x0}px;top:{y0}px;width:{w}px;height:{h}px;"></div>')

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

# -------------------- PDF GENERATOR (Playwright + fallback) --------------------
class PDFGenerator:
    def __init__(self, json_data, output_pdf, original_pdf_path=None):
        if isinstance(json_data, str):
            with open(json_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = json_data
        self.output_pdf = output_pdf
        self.original_pdf_path = original_pdf_path
        self._register_fonts()

    def _register_fonts(self):
        try:
            for lang, font_path in FONTS.items():
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(lang.capitalize(), font_path))
        except Exception:
            pass

    def generate_pdf(self):
        pages = self.data.get("pages", [])
        if not pages:
            raise RuntimeError("No pages in JSON")

        lang = detect_language(self.data)
        temp_pdfs = []
        merger = PdfMerger()

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                for i, page_data in enumerate(pages):
                    html_content = build_page_html(page_data, lang)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as fh:
                        fh.write(html_content)
                        html_path = fh.name
                    page = browser.new_page()
                    page.set_viewport_size({"width": 595, "height": int(_page_height_from(page_data))})
                    page.goto(f"file://{html_path}")
                    page.wait_for_timeout(700)
                    tmp_pdf = f"temp_page_{i}.pdf"
                    page.pdf(path=tmp_pdf, width=f"{595}px", height=f"{int(_page_height_from(page_data))}px", margin={"top":"0","right":"0","bottom":"0","left":"0"}, print_background=True)
                    merger.append(tmp_pdf)
                    temp_pdfs.append(tmp_pdf)
                    page.close()
                    os.unlink(html_path)
                browser.close()

                # if original provided, overlay generated pages onto original
                if self.original_pdf_path and os.path.exists(self.original_pdf_path):
                    reader = PdfReader(self.original_pdf_path)
                    writer = PdfWriter()
                    for idx, base_page in enumerate(reader.pages):
                        overlay_page = None
                        if idx < len(temp_pdfs):
                            overlay_reader = PdfReader(temp_pdfs[idx])
                            overlay_page = overlay_reader.pages[0]
                        if overlay_page:
                            base_page.merge_page(overlay_page)
                        writer.add_page(base_page)
                    with open(self.output_pdf, "wb") as out_f:
                        writer.write(out_f)
                else:
                    merger.write(self.output_pdf)
                    merger.close()

        except Exception as e:
            # fallback to ReportLab drawing (page-by-page)
            self.generate_pdf_fallback()
        finally:
            for tp in temp_pdfs:
                if os.path.exists(tp):
                    os.unlink(tp)

    def generate_pdf_fallback(self):
        pages = self.data.get("pages", [])
        pdf = canvas.Canvas(self.output_pdf, pagesize=(595, 842))
        for idx, pd in enumerate(pages):
            if idx > 0:
                pdf.showPage()
            # white background to cover originals
            pdf.setFillColor(colors.white)
            pdf.rect(0, 0, 595, 842, stroke=0, fill=1)
            self.draw_images(pdf, pd)
            self.draw_text(pdf, pd)
            self.draw_layout_elements(pdf, pd)
            self.draw_vector_symbols(pdf, pd)
        pdf.save()

    def draw_images(self, pdf, page_data):
        page_h = _page_height_from(page_data)
        for img in page_data.get("images", []):
            pos = img['position']
            x0 = pos['x0']
            h = pos['height']
            y0 = _top_from_bottom(pos['y0'], h, page_h)
            w = pos['width']
            data = img.get('data')
            if not data:
                continue
            try:
                img_bytes = base64.b64decode(data)
                pdf.drawImage(ImageReader(BytesIO(img_bytes)), x0, y0, w, h)
            except Exception:
                pass

    def draw_text(self, pdf, page_data):
        page_h = _page_height_from(page_data)
        # ONLY ALLOWED CONTENT TYPES
        allowed_content_types = ["question", "exam_reference", "normal_text", "header", "instruction"]
        
        for block in page_data.get("text_content", []):
            content_type = block.get("content_type", "")
            if content_type not in allowed_content_types:
                continue
                
            translated = block.get("translated_content", "") or ""
            original = block.get("content", "") or ""
            content = translated if translated.strip() else original
            if not content:
                continue
            x0 = block['position']['x0']
            font_size = block.get('font', {}).get('size', 12)
            y0 = _top_from_bottom(block['position']['y0'], font_size, page_h)
            try:
                lang = detect_language(self.data)
                fname = lang.capitalize()
                pdf.setFont(fname, font_size)
            except Exception:
                pdf.setFont("Helvetica", font_size)
            # draw white rect then text to hide original
            safe = clean(content)
            try:
                text_w = pdf.stringWidth(safe, pdf._fontname, font_size)
            except Exception:
                text_w = max(10, len(safe) * (font_size * 0.5))
            pdf.setFillColor(colors.white)
            pdf.rect(x0-2, y0-(font_size*0.25)-2, text_w+4, font_size+4, stroke=0, fill=1)
            pdf.setFillColor(colors.black)
            pdf.drawString(x0, y0, safe)

    def draw_layout_elements(self, pdf, page_data):
        page_h = _page_height_from(page_data)
        for el in page_data.get("layout_elements", []):
            if el.get("type") == "line":
                x0 = el['position']['x0']; y0b = el['position']['y0']
                x1 = el['position']['x1']; y1b = el['position']['y1']
                y0 = _top_from_bottom(y0b, 0, page_h); y1 = _top_from_bottom(y1b, 0, page_h)
                pdf.setStrokeColor(colors.black); pdf.setLineWidth(0.5)
                pdf.line(x0, y0, x1, y1)

    def draw_vector_symbols(self, pdf, page_data):
        page_h = _top_from_bottom(page_data)
        for s in page_data.get("vector_symbols", []):
            x0, y0o, x1, y1o = s['bbox']
            h = y1o - y0o
            y0 = _top_from_bottom(y0o, h, page_h)
            pdf.setStrokeColor(colors.black); pdf.setLineWidth(1)
            pdf.rect(x0, y0, x1 - x0, h, stroke=1, fill=0)

# -------------------- OVERLAY PDF GENERATOR (SAFE COVER) --------------------
class OverlayPDFGenerator:
    def __init__(self, json_data, original_pdf_path, output_pdf):
        if isinstance(json_data, str):
            with open(json_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = json_data
        self.original_pdf_path = original_pdf_path
        self.output_pdf = output_pdf
        self.font_name_map = {}
        self._register_fonts()

    def _register_fonts(self):
        try:
            for lang, font_path in FONTS.items():
                if os.path.exists(font_path):
                    name = f"Font_{lang}"
                    pdfmetrics.registerFont(TTFont(name, font_path))
                    self.font_name_map[lang] = name
        except Exception:
            pass

    def _clean(self, s):
        return clean(s)

    def _detect_language(self):
        return detect_language(self.data)

    def generate_pdf(self):
        if not os.path.exists(self.original_pdf_path):
            raise FileNotFoundError(self.original_pdf_path)
        reader = PdfReader(self.original_pdf_path)
        writer = PdfWriter()
        lang = self._detect_language()
        pages = self.data.get("pages", [])
        
        # ONLY ALLOWED CONTENT TYPES
        allowed_content_types = ["question", "exam_reference", "normal_text", "header", "instruction"]
        
        for i, page_data in enumerate(pages):
            if i >= len(reader.pages):
                continue
            base = reader.pages[i]
            dims = page_data.get("dimensions", {})
            w = float(dims.get("width", 595) or 595)
            h = float(dims.get("height", 842) or 842)
            packet = io.BytesIO()
            c = canvas.Canvas(packet, pagesize=(w, h))
            blocks_drawn = 0
            
            for block in page_data.get("text_content", []):
                # CHECK CONTENT TYPE - ONLY PROCESS ALLOWED TYPES
                content_type = block.get("content_type", "")
                if content_type not in allowed_content_types:
                    continue
                    
                translated = block.get("translated_content", "") or ""
                original = block.get("content", "") or ""
                content = translated if translated.strip() else original
                if not content:
                    continue
                pos = block.get("position", {})
                x0 = float(pos.get("x0", 0) or 0)
                y0_json = float(pos.get("y0", 0) or 0)
                font_size = float(block.get("font", {}).get("size", 12) or 12)
                # convert
                if y0_json > h * 0.9:
                    y0 = h - y0_json - font_size
                else:
                    y0 = y0_json
                if y0 < 0: y0 = 0
                if y0 > h - font_size: y0 = h - font_size
                use_translated = bool(translated.strip())
                if use_translated and lang in self.font_name_map:
                    fname = self.font_name_map[lang]
                    try:
                        c.setFont(fname, font_size)
                    except Exception:
                        c.setFont("Helvetica", font_size)
                        fname = "Helvetica"
                else:
                    fname = "Helvetica"
                    c.setFont(fname, font_size)
                safe = self._clean(content)
                lines = safe.splitlines() or [safe]
                lh = font_size * 1.15
                pad = max(2, font_size * 0.15)
                cur_y = y0
                for ln in lines:
                    try:
                        tw = c.stringWidth(ln or " ", fname, font_size)
                    except Exception:
                        tw = max(10, len(ln) * (font_size * 0.5))
                    descent = font_size * 0.2
                    rx = x0 - pad; ry = cur_y - descent - pad; rw = tw + pad * 2; rh = font_size + pad * 2
                    if rx < 0:
                        rw += rx; rx = 0
                    if ry < 0:
                        rh += ry; ry = 0
                    c.setFillColor(colors.white); c.rect(rx, ry, rw, rh, stroke=0, fill=1)
                    c.setFillColor(colors.black); c.drawString(x0, cur_y, ln or " ")
                    cur_y -= lh
                blocks_drawn += 1
            c.save(); packet.seek(0)
            try:
                overlay = PdfReader(packet).pages[0]
                try:
                    base.merge_page(overlay)
                except Exception:
                    try:
                        base.mergePage(overlay)
                    except Exception:
                        pass
                writer.add_page(base)
            except Exception:
                writer.add_page(base)
        with open(self.output_pdf, "wb") as out_f:
            writer.write(out_f)

# -------------------- CLI / Example --------------------
if __name__ == "__main__":
    JSON_PATH = r"E:\PDFExtraction\translated_jsons\sbi_extracted_hi.json"
    ORIGINAL_PDF = r"E:\PDFExtraction\SBI Clerk Prelims.pdf"
    OUTPUT_OVERLAY = "output_translated_overlay.pdf"
    # Use OverlayPDFGenerator (recommended)
    if not os.path.exists(JSON_PATH):
        print("JSON not found:", JSON_PATH)
    elif not os.path.exists(ORIGINAL_PDF):
        print("Original PDF not found:", ORIGINAL_PDF)
    else:
        gen = OverlayPDFGenerator(JSON_PATH, ORIGINAL_PDF, OUTPUT_OVERLAY)
        gen.generate_pdf()
        print("Done:", OUTPUT_OVERLAY)