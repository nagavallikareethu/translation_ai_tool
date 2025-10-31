"""
COMPLETE PDF PROCESSING PIPELINE
=================================
1. PDF → JSON Extraction (with enhanced OCR and vector detection)
2. JSON Translation (multiple Indian languages with smart filtering)  
3. PDF Generation (faithful layout reconstruction)
"""

import pdfplumber
import json
import base64
from pathlib import Path
from typing import Dict, List, Any
import io
import fitz  # PyMuPDF
import re
from datetime import datetime
import time
import os
import copy
import math
import pathlib
import tempfile
from io import BytesIO
from PyPDF2 import PdfMerger
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from playwright.sync_api import sync_playwright
import html
from deep_translator import GoogleTranslator

# ============================================================
# CONFIGURATION - UPDATED WITH PROPER ODIA FONT
# ============================================================
# Get script directory for relative font paths
SCRIPT_DIR = Path(__file__).parent.resolve()
FONTS_DIR = SCRIPT_DIR / "fonts"

FONTS = {
    "telugu": str(FONTS_DIR / "NotoSansTelugu-Regular.ttf"),
    "hindi": str(FONTS_DIR / "TiroDevanagariHindi-Regular.ttf"), 
    "odia": str(FONTS_DIR / "NotoSansOriya-Regular.ttf")  # Updated Odia font path
}

# Alternative Odia fonts you can use:
# - "NotoSansOriya-Regular.ttf" (Google's Noto Sans - recommended)
# - "Arial_Unicode_MS.ttf" (contains Odia script)
# - "Samyak-Oriya.ttf" (specific Odia font)

LANG_OPTIONS = {
    "1": ("hi", "Hindi"),
    "2": ("te", "Telugu"), 
    "3": ("or", "Odia"),
    "4": ("ta", "Tamil"),
    "5": ("ml", "Malayalam"),
    "6": ("bn", "Bengali"),
    "7": ("gu", "Gujarati"),
    "8": ("pa", "Punjabi"),
    "9": ("mr", "Marathi")
}

MAX_RETRIES = 3
RETRY_DELAY = 2

# ============================================================
# ODIA LANGUAGE SPECIFIC SUPPORT
# ============================================================
class OdiaSupport:
    """Special handling for Odia language and script"""
    
    # Odia Unicode range: 0x0B00-0x0B7F
    ODIA_UNICODE_RANGE = r'[\u0B00-\u0B7F]'
    
    @staticmethod
    def is_odia_text(text: str) -> bool:
        """Check if text contains Odia characters"""
        return bool(re.search(OdiaSupport.ODIA_UNICODE_RANGE, text))
    
    @staticmethod
    def get_odia_font_path():
        """Get Odia font path with fallbacks"""
        # Use relative paths from script directory
        script_dir = Path(__file__).parent.resolve()
        fonts_dir = script_dir / "fonts"
        
        possible_paths = [
            fonts_dir / "NotoSansOriya-Regular.ttf",
            fonts_dir / "NotoSansOriya.ttf",
            fonts_dir / "AnekOdia-Regular.ttf",  # Alternative Odia font
            Path("C:/Windows/Fonts/Nirmala.ttf") if os.name == "nt" else None,  # Windows font
            Path("C:/Windows/Fonts/Arial.ttf") if os.name == "nt" else None,    # Fallback
        ]
        
        for path in possible_paths:
            if path and path.exists():
                return str(path)
        return None

# Update the font configuration with proper Odia support
def setup_fonts():
    """Setup fonts with proper Odia support"""
    fonts_config = FONTS.copy()
    
    # Ensure Odia font exists, if not try to find alternatives
    if not os.path.exists(fonts_config.get("odia", "")):
        odia_font = OdiaSupport.get_odia_font_path()
        if odia_font:
            fonts_config["odia"] = odia_font
            print(f"✅ Using Odia font: {odia_font}")
        else:
            print("⚠️  Odia font not found. Using Telugu font as fallback.")
            fonts_config["odia"] = fonts_config["telugu"]
    
    return fonts_config

# Initialize fonts
FONTS = setup_fonts()

# ============================================================
# UNIFIED COORDINATE SYSTEM
# ============================================================
class CoordinateConverter:
    """Convert all coordinates to consistent system (Y=0 at bottom)"""
    
    def __init__(self, page_height=842):
        self.page_height = page_height
    
    def pdfplumber_to_standard(self, y):
        """Convert pdfplumber coordinates (Y=0 at top) to standard (Y=0 at bottom)"""
        return self.page_height - y
    
    def pymupdf_to_standard(self, y, height=0):
        """Convert PyMuPDF coordinates to standard"""
        return self.page_height - y - height

# ============================================================
# ENHANCED TEXT CLEANING
# ============================================================
def enhanced_clean_extracted_text(text: str) -> str:
    """Comprehensive OCR correction while preserving special content"""
    if not text:
        return text
    
    # Preserve Telugu text - DO NOT modify
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    if telugu_chars:
        return text
    
    # Preserve Odia text - DO NOT modify
    odia_chars = re.findall(r'[\u0B00-\u0B7F]', text)
    if odia_chars:
        return text
    
    # Enhanced OCR corrections
    replacements = {
        'wi√': 'will', 'fi√': 'fill', 'studesnts': 'students',
        'shk': 'CS', '√': 'square root', '÷': '÷', '×': '×',
        '²': '²', '³': '³', '°': '°', 'ﬁ': 'fi', 'ﬂ': 'fl'
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Fix common OCR issues with regex
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix spaced numbers
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)  # Fix hyphenated words
    
    return text.strip()

def merge_fragmented_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge single-character text blocks into proper words"""
    if not blocks:
        return blocks
    
    merged_blocks = []
    current_block = None
    
    for block in blocks:
        content = block.get('content', '').strip()
        
        # Check if this is a fragmented character (single char with small font)
        is_fragment = (len(content) == 1 and 
                      block.get('font', {}).get('size', 12) < 8 and
                      content.isalpha())
        
        if is_fragment:
            if current_block:
                # Merge with current block
                current_block['content'] += content
                # Update position to include the merged character
                current_block['position']['x1'] = block['position']['x1']
                current_block['position']['width'] = current_block['position']['x1'] - current_block['position']['x0']
                current_block['word_count'] = len(current_block['content'].split())
            else:
                # Start new merged block
                current_block = block.copy()
        else:
            # Non-fragment block
            if current_block:
                merged_blocks.append(current_block)
                current_block = None
            merged_blocks.append(block)
    
    # Don't forget the last merged block
    if current_block:
        merged_blocks.append(current_block)
    
    return merged_blocks

def improved_content_classification(text: str) -> str:
    """Better content classification for government exam papers"""
    # Check for Telugu text first (preserve it)
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    if telugu_chars:
        return "telugu_text"
    
    # Check for Odia text (preserve it)
    odia_chars = re.findall(r'[\u0B00-\u0B7F]', text)
    if odia_chars:
        return "odia_text"
    
    text_lower = text.lower().strip()
    
    # Instructions
    if any(word in text_lower for word in ['directions', 'study the', 'answer the', 'following questions', 'instruction']):
        return "instruction"
    
    # Questions (starts with number and period)
    elif re.match(r'^\d+\.', text.strip()):
        return "question"
    
    # Options (starts with number and parenthesis)
    elif re.match(r'^[1-5]\)', text.strip()):
        return "option"
    
    # Data labels (years, numbers on axes)
    elif re.match(r'^(201[2-9]|202[0-9]|60|50|40|30|20|10|0)$', text.strip()):
        return "data_label"
    
    # Headers (all caps or title case)
    elif re.match(r'^[A-Z][A-Z\s]+$', text.strip()) or text_lower in ['numerical ability', 'years', 'question', 'answer']:
        return "header"
    
    # Mathematical content
    elif any(symbol in text for symbol in ['÷', '×', '=', '%', '√', '²', '³', 'square root']):
        return "mathematical"
    
    # Exam names and acronyms
    elif any(acronym in text for acronym in ['SBI', 'LIC', 'NIACL', 'MTS', 'CGL', 'CHSL', 'RRB', 'IBPS']):
        return "exam_reference"
    
    else:
        return "normal_text"

# ============================================================
# ENHANCED BAR CHART DETECTION
# ============================================================
class EnhancedBarChartDetector:
    def __init__(self):
        self.chart_keywords = [
            'bar chart', 'students registered', 'exams', 'years', 'data interpretation',
            'vertical bar', 'horizontal bar', 'column chart', 'graph', 'diagram'
        ]
    
    def detect_bar_chart_enhanced(self, image_info, page_text_blocks, page_num):
        """Enhanced bar chart detection with contextual analysis"""
        detection_score = 0
        detection_features = []
        
        # Feature 1: Image dimension analysis (30% weight)
        width = image_info['dimensions']['width']
        height = image_info['dimensions']['height']
        aspect_ratio = width / height if height > 0 else 0
        
        if 1.2 <= aspect_ratio <= 3.0:  # Typical bar chart ratios
            detection_score += 0.3
            detection_features.append('optimal_aspect_ratio')
        
        # Feature 2: Contextual text analysis (40% weight)
        contextual_score = self._analyze_surrounding_context(page_text_blocks)
        detection_score += contextual_score * 0.4
        if contextual_score > 0.5:
            detection_features.append('strong_contextual_evidence')
        
        # Feature 3: Size-based confidence (30% weight)
        if width > 300 and height > 150:  # Significant chart size
            detection_score += 0.3
            detection_features.append('adequate_size')
        
        # Feature 4: Page position analysis (bonus)
        if self._is_typical_chart_position(image_info['position']):
            detection_score += 0.1
            detection_features.append('typical_position')
        
        return {
            'is_bar_chart_candidate': detection_score >= 0.6,
            'confidence': min(detection_score, 1.0),
            'detection_features': detection_features,
            'chart_type': self._classify_chart_type(aspect_ratio, contextual_score),
            'extraction_priority': 'high' if detection_score > 0.7 else 'medium'
        }
    
    def _analyze_surrounding_context(self, text_blocks):
        """Analyze surrounding text for chart indicators"""
        context_score = 0
        relevant_text = ""
        
        for block in text_blocks:
            content = block.get('content', '').lower()
            relevant_text += " " + content
            
            # Direct chart mentions
            if any(keyword in content for keyword in self.chart_keywords):
                context_score += 0.3
            
            # Data interpretation patterns
            if any(pattern in content for pattern in ['study the', 'following data', 'answer the questions']):
                context_score += 0.2
            
            # Year patterns (common in charts)
            if re.search(r'20[0-9]{2}', content):
                context_score += 0.1
        
        # Check for exam acronyms in context
        if any(acronym in relevant_text for acronym in ['mts', 'cgl', 'chsl']):
            context_score += 0.2
        
        return min(context_score, 1.0)
    
    def _is_typical_chart_position(self, position):
        """Check if image is in typical chart position"""
        x0, y0 = position.get('x0', 0), position.get('y0', 0)
        # Charts are typically centered or top-half of page
        return x0 > 100 and y0 > 200
    
    def _classify_chart_type(self, aspect_ratio, contextual_score):
        """Classify chart type based on features"""
        if aspect_ratio > 2.0:
            return "vertical_bar_chart"
        elif aspect_ratio < 1.0:
            return "horizontal_bar_chart"
        else:
            return "grouped_bar_chart"

def extract_axis_labels_from_context(page_text_blocks, chart_position):
    """Extract axis labels from surrounding text blocks"""
    axis_data = {'x_labels': [], 'y_labels': [], 'title': ''}
    
    for block in page_text_blocks:
        block_pos = block.get('position', {})
        content = block.get('content', '').strip()
        
        # Check if text block is near the chart (simplified proximity check)
        chart_x, chart_y = chart_position.get('x0', 0), chart_position.get('y0', 0)
        block_x, block_y = block_pos.get('x0', 0), block_pos.get('y0', 0)
        
        if abs(block_x - chart_x) < 200 and abs(block_y - chart_y) < 200:
            
            # Extract X-axis labels (years: 2012, 2013, etc.)
            year_matches = re.findall(r'20[0-9]{2}', content)
            if year_matches:
                axis_data['x_labels'] = sorted(set(year_matches))
                print(f"📅 Extracted X-axis years: {axis_data['x_labels']}")
            
            # Extract Y-axis labels (numbers: 0, 10, 20, etc.)
            number_matches = re.findall(r'\b\d{1,3}\b', content)
            if len(number_matches) >= 3 and all(int(n) % 10 == 0 for n in number_matches[:3]):
                axis_data['y_labels'] = sorted(set(int(n) for n in number_matches))
                print(f"📊 Extracted Y-axis scale: {axis_data['y_labels']}")
            
            # Extract chart title
            if len(content) > 20 and len(content) < 100 and not content.startswith('1)'):
                axis_data['title'] = content
                print(f"🏷️ Extracted chart title: {content[:50]}...")
    
    return axis_data

# ============================================================
# ENHANCED ERROR HANDLING
# ============================================================
class GracefulDegradation:
    def __init__(self):
        self.fallback_strategies = {
            'text_extraction': self._fallback_text_extraction,
            'image_processing': self._fallback_image_processing,
            'table_parsing': self._fallback_table_parsing
        }
        self.error_log = []
    
    def handle_extraction_error(self, operation, error, context):
        """Comprehensive error handling with graceful degradation"""
        error_id = f"ERR_{hash(str(error))[:8]}"
        
        # Log error with context
        self._log_error(error_id, operation, error, context)
        
        # Try fallback strategy
        fallback_result = self._attempt_fallback(operation, context)
        
        if fallback_result['success']:
            return {
                'success': True,
                'data': fallback_result['data'],
                'quality': 'degraded',
                'error_handled': error_id,
                'fallback_used': fallback_result['method']
            }
        else:
            return {
                'success': False,
                'error': f"Operation failed: {str(error)}",
                'error_id': error_id,
                'suggested_action': self._get_recovery_suggestion(operation)
            }
    
    def _attempt_fallback(self, operation, context):
        """Attempt fallback strategies for failed operations"""
        fallback_method = self.fallback_strategies.get(operation)
        
        if fallback_method:
            try:
                result = fallback_method(context)
                return {'success': True, 'data': result, 'method': operation}
            except Exception as fallback_error:
                print(f"⚠️ Fallback also failed: {fallback_error}")
        
        return {'success': False, 'method': 'none_available'}
    
    def _fallback_text_extraction(self, context):
        """Fallback for text extraction failures"""
        print("🔄 Using fallback text extraction...")
        # Simplified text extraction as fallback
        return {"text_blocks": [], "method": "fallback_simple"}
    
    def _fallback_image_processing(self, context):
        """Fallback for image processing failures"""
        print("🔄 Using fallback image processing...")
        return {"images": [], "method": "fallback_basic"}
    
    def _fallback_table_parsing(self, context):
        """Fallback for table parsing failures"""
        print("🔄 Using fallback table parsing...")
        return {"tables": [], "method": "fallback_simple"}
    
    def _log_error(self, error_id, operation, error, context):
        """Log error with context"""
        error_entry = {
            'error_id': error_id,
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': str(context)[:200]  # Limit context length
        }
        self.error_log.append(error_entry)
        print(f"📝 Error logged: {error_id} - {operation} - {error}")
    
    def _get_recovery_suggestion(self, operation):
        """Get recovery suggestions based on operation type"""
        suggestions = {
            'text_extraction': 'Try adjusting OCR settings or preprocessing PDF',
            'image_processing': 'Reduce image quality or skip image extraction',
            'table_parsing': 'Use simpler table detection algorithm',
            'default': 'Check PDF quality and try alternative extraction methods'
        }
        return suggestions.get(operation, suggestions['default'])
    
    def generate_error_report(self):
        """Generate error analytics report"""
        if not self.error_log:
            return {"status": "No errors recorded", "health_score": 1.0}
        
        total_errors = len(self.error_log)
        error_types = {}
        
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": total_errors,
            "error_distribution": error_types,
            "health_score": max(0, 1 - (total_errors / 100)),
            "recent_errors": self.error_log[-5:] if self.error_log else []
        }

# ============================================================
# VECTOR MATH SYMBOL DETECTION
# ============================================================
def detect_math_drawings(pdf_path: str) -> dict:
    """Detect vector-drawn math shapes with CORRECT coordinates"""
    detected = {}
    doc = fitz.open(pdf_path)
    print(f"\n🔍 Scanning {pdf_path} for vector math symbols...")
    
    for page_num, page in enumerate(doc):
        page_height = page.rect.height  # Get actual page height
        drawings = page.get_drawings()
        math_like = []
        
        for d in drawings:
            bbox = d.get("rect", None)
            if not bbox:
                continue
                
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            aspect = width / max(height, 1)
            
            # FIX: Convert PyMuPDF coordinates to standard PDF (Y=0 at bottom)
            fixed_y0 = page_height - bbox[3]  # Convert Y coordinates
            fixed_y1 = page_height - bbox[1]
            
            # Heuristic for math lines / bars / radicals
            if (height < 3 and width > 10) or (aspect < 0.2 and height > 10):
                math_like.append({
                    "type": "vector_symbol",
                    "symbol_guess": "fraction_bar" if height < 3 else "radical_or_vertical",
                    "bbox": [bbox[0], fixed_y0, bbox[2], fixed_y1],  # FIXED coordinates
                    "dimensions": {"width": width, "height": height},
                    "source": "drawing-stroke",
                    "page": page_num
                })
                
        if math_like:
            detected[page_num] = math_like
            print(f"📐 Page {page_num + 1}: {len(math_like)} math symbols with FIXED coordinates")
            
    doc.close()
    return detected

def merge_math_symbols_into_json(json_path: str, detected_math: dict):
    """Merge detected math symbols directly into the same JSON (no new file)"""
    if not detected_math:
        print("ℹ️ No vector math symbols found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for page in data.get("pages", []):
        page_index = page.get("page_number", 1) - 1
        if page_index in detected_math:
            page["vector_symbols"] = detected_math[page_index]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Math symbols merged into JSON: {json_path}")

# ============================================================
# TRANSLATION HELPER FUNCTIONS
# ============================================================
def should_translate_content(text: str, content_type: str = None) -> bool:
    """
    Smart content filtering for government exam papers
    Returns False for content that should NOT be translated
    """
    if not text or not text.strip():
        return False
    
    text_clean = text.strip()
    
    # ENHANCED: Check if content_type might be incorrectly set
    # If text is clearly descriptive English with percentages, override mathematical type
    if content_type == 'mathematical':
        # Check if this is actually descriptive text with percentages
        words = re.findall(r'\b[a-zA-Z]+\b', text_clean)
        if len(words) >= 3:  # If there are at least 3 words, it's likely descriptive text
            # Count percentage symbols
            percent_count = text_clean.count('%')
            # If it's mostly text with some percentages, override the content_type
            if percent_count <= 2 and len(words) > percent_count * 2:
                content_type = 'normal_text'  # Override incorrectly set content_type
    
    # Use content_type from JSON if available (highest priority)
    if content_type and content_type in ['mathematical', 'data_label', 'option', 'coordinate', 'telugu_text', 'odia_text']:
        return False
    
    # Don't translate pure numbers/options (like "1) 445") but allow mixed content
    if re.match(r'^[1-5]\)\s*\d+$', text_clean):
        return False
    
    # Don't translate table headers that are purely numbers
    if re.match(r'^\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\?$', text_clean):
        return False
    
    # Don't translate pure English acronyms that should stay same
    acronyms = ['SBI', 'LIC', 'NIACL', 'MTS', 'CGL', 'CHSL', 'CCE', 'SSC', 'UPSC', 'IBPS']
    if all(word in acronyms or word.isdigit() for word in text_clean.upper().split()):
        return False
    
    # Don't translate single characters UNLESS they are words
    if len(text_clean) <= 1:
        return False
    elif len(text_clean) == 2:
        # Allow common 2-letter words but block pure numbers/symbols
        if text_clean.lower() in ['is', 'am', 'are', 'to', 'of', 'in', 'on', 'at', 'by', 'we', 'us', 'he', 'she', 'it', 'my', 'your', 'our', 'their', 'an', 'as', 'or', 'if', 'so', 'no', 'yes', 'ok']:
            return True
        elif text_clean.isalpha():
            return True
        else:
            return False
    
    # Don't translate pure data labels (years, numbers) but allow them in sentences
    if re.match(r'^(201[2-9]|202[0-9]|60|50|40|30|20|10|0)$', text_clean) and len(text_clean) <= 4:
        return False
    
    # ENHANCED MATH DETECTION: Only block PURE mathematical expressions
    # Keep % symbol since it's common in text descriptions
    math_indicators = ['√', '×', '÷', '²', '³', '∛', '∜', '=']  # Removed '%' and '^' as they appear in text
    
    # Check for complex mathematical expressions (multiple math symbols)
    math_symbols_found = [symbol for symbol in math_indicators if symbol in text_clean]
    
    if math_symbols_found:
        # Count words and math symbols more accurately
        words = re.findall(r'\b[a-zA-Z]+\b', text_clean)  # Only count actual words
        math_symbol_count = len(math_symbols_found)
        
        # If no meaningful words found and multiple math symbols, it's likely pure math
        if len(words) == 0 and math_symbol_count >= 2:
            return False
        
        # Only block if it's primarily math (more than 60% math content)
        if len(words) > 0 and math_symbol_count > max(len(words) * 0.6, 3):
            return False
        # Allow mixed content (text with some math symbols)
        else:
            return True
    
    # ENHANCED: Allow percentage text and other common patterns
    # Check if text contains percentage but is actually descriptive text
    if '%' in text_clean:
        # Count words in the text to determine if it's descriptive
        words = re.findall(r'\b[a-zA-Z]+\b', text_clean)
        if len(words) >= 3:  # If there are at least 3 words, it's likely descriptive text
            return True
    
    return True

def safe_translate(text: str, target_lang: str) -> str:
    """Translate safely with retry and log in console."""
    if not text.strip():
        return text
    
    # Enhanced text cleaning before translation
    text = text.strip()
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
            print(f"   ✅ Translation successful (attempt {attempt})")
            return translated
        except Exception as e:
            print(f"⚠️ Retry {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    
    print(f"❌ All translation attempts failed for: {text[:50]}...")
    return text  # Return original if all attempts fail

def count_json_elements(data):
    """Count all critical elements in JSON for verification"""
    vector_count = 0
    layout_count = 0
    image_count = 0
    table_count = 0
    text_blocks = 0
    translatable_blocks = 0
    
    for page in data.get("pages", []):
        vector_count += len(page.get('vector_symbols', []))
        layout_count += len(page.get('layout_elements', []))
        image_count += len(page.get('images', []))
        table_count += len(page.get('tables', []))
        
        for text_block in page.get('text_content', []):
            text_blocks += 1
            content = text_block.get('content', '')
            content_type = text_block.get('content_type', '')
            if should_translate_content(content, content_type):
                translatable_blocks += 1
    
    return {
        'vector_symbols': vector_count,
        'layout_elements': layout_count,
        'images': image_count,
        'tables': table_count,
        'text_blocks': text_blocks,
        'translatable_blocks': translatable_blocks
    }

def verify_coordinate_preservation(original_data, translated_data):
    """Verify that coordinates are preserved during translation"""
    print(f"\n🎯 COORDINATE PRESERVATION CHECK:")
    
    original_pages = original_data.get("pages", [])
    translated_pages = translated_data.get("pages", [])
    
    if len(original_pages) != len(translated_pages):
        print("❌ Page count mismatch!")
        return False
    
    all_coordinates_preserved = True
    
    for page_idx, (orig_page, trans_page) in enumerate(zip(original_pages, translated_pages)):
        orig_vectors = orig_page.get("vector_symbols", [])
        trans_vectors = trans_page.get("vector_symbols", [])
        
        if len(orig_vectors) != len(trans_vectors):
            print(f"❌ Page {page_idx + 1}: Vector symbol count mismatch")
            all_coordinates_preserved = False
        else:
            for vec_idx, (orig_vec, trans_vec) in enumerate(zip(orig_vectors, trans_vectors)):
                orig_bbox = orig_vec.get("bbox", [])
                trans_bbox = trans_vec.get("bbox", [])
                if orig_bbox != trans_bbox:
                    print(f"❌ Page {page_idx + 1}, Vector {vec_idx + 1}: BBOX coordinates changed")
                    all_coordinates_preserved = False
        
        orig_layouts = orig_page.get("layout_elements", [])
        trans_layouts = trans_page.get("layout_elements", [])
        
        if len(orig_layouts) != len(trans_layouts):
            print(f"❌ Page {page_idx + 1}: Layout element count mismatch")
            all_coordinates_preserved = False
        else:
            for layout_idx, (orig_layout, trans_layout) in enumerate(zip(orig_layouts, trans_layouts)):
                if orig_layout.get("position", {}) != trans_layout.get("position", {}):
                    print(f"❌ Page {page_idx + 1}, Layout {layout_idx + 1}: Position coordinates changed")
                    all_coordinates_preserved = False
        
        orig_text_blocks = orig_page.get("text_content", [])
        trans_text_blocks = trans_page.get("text_content", [])
        
        if len(orig_text_blocks) != len(trans_text_blocks):
            print(f"❌ Page {page_idx + 1}: Text block count mismatch")
            all_coordinates_preserved = False
            continue
        
        for block_idx, (orig_block, trans_block) in enumerate(zip(orig_text_blocks, trans_text_blocks)):
            orig_pos = orig_block.get("position", {})
            trans_pos = trans_block.get("position", {})
            
            # Compare all position coordinates
            if orig_pos != trans_pos:
                print(f"❌ Page {page_idx + 1}, Block {block_idx + 1}: Position coordinates changed!")
                print(f"   Original: {orig_pos}")
                print(f"   Translated: {trans_pos}")
                all_coordinates_preserved = False
    
    if all_coordinates_preserved:
        print("✅ SUCCESS: All coordinates preserved during translation!")
    else:
        print("❌ WARNING: Some coordinates were modified during translation!")
    
    return all_coordinates_preserved

# ============================================================
# PDF TO JSON CONVERTER CLASS
# ============================================================
class PDFToJSONConverter:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.bar_chart_detector = EnhancedBarChartDetector()
        self.error_handler = GracefulDegradation()

    def _is_translatable_content(self, text: str) -> bool:
        """Determine if text block should be translated - ENHANCED VERSION"""
        if not text or len(text.strip()) < 2:
            return False
        
        # Don't translate Telugu text
        telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
        if telugu_chars:
            return False
        
        # Don't translate Odia text
        odia_chars = re.findall(r'[\u0B00-\u0B7F]', text)
        if odia_chars:
            return False
        
        # Don't translate mathematical expressions
        math_indicators = ['√', '×', '÷', '²', '³', '=', '%', 'square root']
        if any(symbol in text for symbol in math_indicators):
            return False
        
        # Don't translate pure numbers/options
        if re.match(r'^[1-5]\)\s*\d+$', text.strip()):
            return False
        
        # Don't translate number sequences
        if re.match(r'^\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\?$', text.strip()):
            return False
        
        # Don't translate acronyms
        acronyms = ['SBI', 'LIC', 'NIACL', 'MTS', 'CGL', 'CHSL', 'CCE', 'RRB', 'IBPS']
        if all(word in acronyms or word.isdigit() for word in text.upper().split()):
            return False
        
        return True

    def _classify_content_type(self, text: str) -> str:
        """Enhanced content classification"""
        return improved_content_classification(text)

    def _extract_text_enhanced(self, page, coord_converter) -> List[Dict[str, Any]]:
        """Enhanced text extraction with consistent coordinates"""
        text_blocks = []
        try:
            words = page.extract_words(
                extra_attrs=["fontname", "size"], 
                x_tolerance=1.5,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True
            )
            
            current_block, current_y, y_tol = [], None, 5
            for w in sorted(words, key=lambda w: (w['top'], w['x0'])):
                w_text = enhanced_clean_extracted_text(w['text'].strip())
                if not w_text:
                    continue
                    
                if current_y is None or abs(w['top'] - current_y) > y_tol:
                    if current_block:
                        text_blocks.append(self._create_text_block_enhanced(current_block, coord_converter))
                    current_block, current_y = [w], w['top']
                else:
                    current_block.append(w)
                    
            if current_block:
                text_blocks.append(self._create_text_block_enhanced(current_block, coord_converter))
                
        except Exception as e:
            print(f"    ⚠️ Enhanced text extraction failed: {e}")
            # Fallback to basic extraction
            try:
                text = page.extract_text()
                if text:
                    cleaned_text = enhanced_clean_extracted_text(text)
                    text_blocks.append({
                        "type": "text_block", 
                        "content": cleaned_text,
                        "position": {"x0": 0, "y0": 0, "x1": float(page.width), "y1": float(page.height)},
                        "font": {"name": "", "size": 0},
                        "word_count": len(cleaned_text.split()),
                        "translation_ready": self._is_translatable_content(cleaned_text),
                        "content_type": self._classify_content_type(cleaned_text),
                        "extraction_method": "fallback_basic"
                    })
            except:
                pass
                
        return text_blocks

    def _create_text_block_enhanced(self, words: List[Dict[str, Any]], coord_converter) -> Dict[str, Any]:
        """Create text block with consistent coordinates"""
        if not words:
            return {}
            
        x0, top = min(w['x0'] for w in words), min(w['top'] for w in words)
        x1, bottom = max(w['x1'] for w in words), max(w['bottom'] for w in words)
        text_content = ' '.join(enhanced_clean_extracted_text(w['text']) for w in words)
        fonts = [w.get('fontname', '') for w in words if w.get('fontname')]
        font_size = words[0].get('size', 0) if words else 0
        
        # Use coordinate converter for consistent coordinates
        fixed_top = coord_converter.pdfplumber_to_standard(bottom)
        fixed_bottom = coord_converter.pdfplumber_to_standard(top)
        
        content_type = self._classify_content_type(text_content)
        should_translate = self._is_translatable_content(text_content)
        
        return {
            "type": "text_block",
            "content": text_content,
            "position": {
                "x0": float(x0), "y0": float(fixed_top),
                "x1": float(x1), "y1": float(fixed_bottom),
                "width": float(x1 - x0), "height": float(bottom - top)
            },
            "font": {
                "name": max(set(fonts), key=fonts.count) if fonts else "", 
                "size": float(font_size),
                "bold": any('bold' in f.lower() for f in fonts),
                "italic": any('italic' in f.lower() for f in fonts)
            },
            "word_count": len(words),
            "translation_ready": should_translate,
            "content_type": content_type,
            "extraction_method": "enhanced"
        }

    def _extract_tables_enhanced(self, page, coord_converter) -> List[Dict[str, Any]]:
        """Extract tables with better structure detection"""
        tables = []
        try:
            table_objects = page.find_tables()
            extracted_tables = page.extract_tables()
            
            for i, (table_obj, table_data) in enumerate(zip(table_objects, extracted_tables)):
                if table_data and any(any(cell for cell in row) for row in table_data):
                    # Convert coordinates
                    fixed_y0 = coord_converter.pdfplumber_to_standard(table_obj.bbox[3])
                    fixed_y1 = coord_converter.pdfplumber_to_standard(table_obj.bbox[1])
                    
                    has_header = self._detect_table_header(table_data)
                    
                    tables.append({
                        "table_number": i + 1,
                        "position": {
                            "x0": float(table_obj.bbox[0]), "y0": float(fixed_y0),
                            "x1": float(table_obj.bbox[2]), "y1": float(fixed_y1)
                        },
                        "data": table_data,
                        "structure": {
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "has_header": has_header,
                            "header_row": 0 if has_header else None
                        },
                        "translation_ready": not has_header,
                        "extraction_method": "enhanced"
                    })
                    
        except Exception as e:
            print(f"    ⚠️ Enhanced table extraction failed: {e}")
            
        return tables

    def _detect_table_header(self, table_data):
        """Detect if table has a header row"""
        if not table_data or len(table_data) < 2:
            return False
        
        first_row = table_data[0]
        second_row = table_data[1] if len(table_data) > 1 else []
        
        # Simple heuristic: header often has text while data has numbers
        first_row_has_text = any(cell and not cell.replace('.', '').isdigit() for cell in first_row if cell)
        second_row_has_numbers = any(cell and cell.replace('.', '').isdigit() for cell in second_row if cell)
        
        return first_row_has_text and second_row_has_numbers

    def _extract_images_enhanced(self, page, include_images, image_handling, 
                                page_num, fitz_doc, coord_converter):
        """Enhanced image extraction with real coordinates and robust error handling"""
        images = []
        if not include_images or not fitz_doc:
            return images
        
        try:
            pymupdf_page = fitz_doc[page_num]
            page_height = pymupdf_page.rect.height
            image_list = pymupdf_page.get_images(full=True)
            
            print(f"    📷 Enhanced image extraction: found {len(image_list)} images")
            
            successful_extractions = 0
            failed_extractions = 0
            
            for idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    
                    # Enhanced error handling for image extraction
                    try:
                        base_image = fitz_doc.extract_image(xref)
                        if not base_image:
                            print(f"    ⚠️ Image {idx + 1}: No image data returned")
                            failed_extractions += 1
                            continue
                            
                        img_bytes = base_image.get("image")
                        if not img_bytes:
                            print(f"    ⚠️ Image {idx + 1}: Empty image bytes")
                            failed_extractions += 1
                            continue
                            
                    except Exception as extract_error:
                        print(f"    ⚠️ Image {idx + 1}: Extraction failed - {extract_error}")
                        failed_extractions += 1
                        continue
                    
                    # Get image properties with fallbacks
                    img_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    if width == 0 or height == 0:
                        print(f"    ⚠️ Image {idx + 1}: Invalid dimensions {width}x{height}")
                        failed_extractions += 1
                        continue
                    
                    # Get image position with enhanced error handling
                    try:
                        img_rect = pymupdf_page.get_image_bbox(img)
                        x0, y0, x1, y1 = img_rect
                        
                        # Validate coordinates
                        if x1 <= x0 or y1 <= y0:
                            print(f"    ⚠️ Image {idx + 1}: Invalid bounding box {x0},{y0},{x1},{y1}")
                            # Use fallback position
                            x0, y0 = 50 + (idx * 10), 50 + (idx * 10)
                            x1, y1 = x0 + width, y0 + height
                            
                    except Exception as coord_error:
                        print(f"    ⚠️ Image {idx + 1}: Coordinate extraction failed - {coord_error}")
                        # Use fallback position
                        x0, y0 = 50 + (idx * 10), 50 + (idx * 10)
                        x1, y1 = x0 + width, y0 + height
                    
                    # Convert coordinates to standard system
                    fixed_y0 = coord_converter.pymupdf_to_standard(y0, height)
                    fixed_y1 = coord_converter.pymupdf_to_standard(y1, height)
                    
                    image_info = {
                        "image_number": idx + 1,
                        "position": {
                            "x0": float(x0), 
                            "y0": float(fixed_y0),
                            "x1": float(x1),
                            "y1": float(fixed_y1),
                            "width": float(width), 
                            "height": float(height)
                        },
                        "dimensions": {"width": float(width), "height": float(height)},
                        "extraction_method": "enhanced_pymupdf",
                        "page": page_num,
                        "image_format": img_ext,
                        "image_size_bytes": len(img_bytes)
                    }
                    
                    # Enhanced bar chart detection
                    try:
                        page_text_blocks = self._extract_text_enhanced(page, coord_converter)
                        chart_detection = self.bar_chart_detector.detect_bar_chart_enhanced(
                            image_info, page_text_blocks, page_num
                        )
                        
                        axis_data = {}
                        if chart_detection['is_bar_chart_candidate']:
                            axis_data = extract_axis_labels_from_context(page_text_blocks, image_info['position'])
                        
                        image_info.update({
                            "is_bar_chart_candidate": chart_detection['is_bar_chart_candidate'],
                            "bar_chart_confidence": chart_detection['confidence'],
                            "chart_type": chart_detection['chart_type'],
                            "detection_features": chart_detection['detection_features'],
                            "extracted_axis_data": axis_data,
                            "description": f"Image {idx+1} ({width}x{height})" + 
                                          (" [BAR CHART]" if chart_detection['is_bar_chart_candidate'] else "")
                        })
                    except Exception as chart_error:
                        print(f"    ⚠️ Chart detection failed for image {idx + 1}: {chart_error}")
                        image_info.update({
                            "is_bar_chart_candidate": False,
                            "bar_chart_confidence": 0.0,
                            "chart_type": "unknown",
                            "detection_features": ["chart_detection_failed"]
                        })
                    
                    # Handle image data based on user preference
                    try:
                        if image_handling == "base64":
                            b64 = base64.b64encode(img_bytes).decode("utf-8")
                            image_info["data"] = b64
                            image_info["image_size_kb"] = len(img_bytes) / 1024
                        elif image_handling == "external":
                            img_filename = f"page_{page_num+1}_img_{idx+1}.{img_ext}"
                            img_path = Path("extracted_images") / img_filename
                            img_path.parent.mkdir(exist_ok=True)
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            image_info["external_path"] = str(img_path)
                            image_info["image_size_kb"] = len(img_bytes) / 1024
                        else:  # metadata only
                            image_info["image_size_kb"] = len(img_bytes) / 1024
                            
                    except Exception as data_error:
                        print(f"    ⚠️ Image data handling failed for image {idx + 1}: {data_error}")
                        # Continue with metadata only
                        image_info["image_size_kb"] = len(img_bytes) / 1024
                        image_info["data_handling_error"] = str(data_error)
                    
                    images.append(image_info)
                    successful_extractions += 1
                    
                    if image_info.get('is_bar_chart_candidate', False):
                        print(f"    ✅ ENHANCED BAR CHART: Image {idx + 1} - Confidence: {image_info['bar_chart_confidence']:.2f}")
                    else:
                        print(f"    ✅ Extracted image {idx + 1} ({width}x{height}) at position {x0:.1f}, {y0:.1f}")
                            
                except Exception as e:
                    print(f"    ⚠️ Enhanced image extraction failed for image {idx + 1}: {e}")
                    failed_extractions += 1
                    
            # Summary for the page
            if successful_extractions > 0:
                print(f"    📊 Image extraction summary: {successful_extractions} successful, {failed_extractions} failed")
            else:
                print(f"    ❌ No images successfully extracted from page {page_num + 1}")
                # Try fallback extraction
                fallback_images = self._extract_images_fallback(page, include_images, image_handling, page_num, fitz_doc, coord_converter)
                images.extend(fallback_images)
                        
        except Exception as e:
            print(f"    ❌ Enhanced image extraction failed on page {page_num + 1}: {e}")
            
        return images

    def _extract_images_fallback(self, page, include_images, image_handling, 
                               page_num, fitz_doc, coord_converter):
        """Fallback image extraction using alternative methods"""
        images = []
        if not include_images or not fitz_doc:
            return images
        
        try:
            pymupdf_page = fitz_doc[page_num]
            
            # Alternative approach: extract images as PDF pixmaps
            pix = pymupdf_page.get_pixmap(matrix=fitz.Matrix(1, 1))
            if pix:
                image_info = {
                    "image_number": 1,
                    "position": {
                        "x0": 0, 
                        "y0": 0,
                        "width": float(pix.width), 
                        "height": float(pix.height)
                    },
                    "dimensions": {"width": float(pix.width), "height": float(pix.height)},
                    "extraction_method": "fallback_pixmap",
                    "page": page_num,
                    "description": "Full page raster extraction (fallback)"
                }
                
                if image_handling == "base64":
                    img_bytes = pix.tobytes("png")
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    image_info["data"] = b64
                    image_info["image_size_kb"] = len(img_bytes) / 1024
                    
                images.append(image_info)
                print(f"    ✅ Fallback: Extracted page as image ({pix.width}x{pix.height})")
                
        except Exception as e:
            print(f"    ⚠️ Fallback image extraction failed: {e}")
            
        return images

    def _extract_layout_enhanced(self, page, coord_converter):
        """Extract layout elements with consistent coordinates"""
        elements = []
        
        try:
            # Extract lines
            for line in page.lines:
                fixed_y0 = coord_converter.pdfplumber_to_standard(line['top'])
                fixed_y1 = coord_converter.pdfplumber_to_standard(line['bottom'])
                
                elements.append({
                    "type": "line",
                    "position": {
                        "x0": float(line['x0']), "y0": float(fixed_y0),
                        "x1": float(line['x1']), "y1": float(fixed_y1)
                    },
                    "width": float(line['width']), 
                    "height": float(line['height']),
                    "extraction_method": "enhanced"
                })
                
            # Extract curves (if any)
            for curve in page.curves:
                elements.append({
                    "type": "curve",
                    "points": curve.get('points', []),
                    "extraction_method": "enhanced"
                })
                
        except Exception as e:
            print(f"    ⚠️ Enhanced layout extraction failed: {e}")
            
        return elements

    def _extract_vector_symbols_enhanced(self, fitz_doc, page_num, coord_converter):
        """Extract vector symbols with consistent coordinates"""
        symbols = []
        
        try:
            page = fitz_doc[page_num]
            page_height = page.rect.height
            drawings = page.get_drawings()
            
            for d in drawings:
                bbox = d.get("rect", None)
                if not bbox:
                    continue
                    
                width = abs(bbox[2] - bbox[0])
                height = abs(bbox[3] - bbox[1])
                aspect = width / max(height, 1)
                
                # Convert coordinates
                fixed_y0 = coord_converter.pymupdf_to_standard(bbox[3], 0)
                fixed_y1 = coord_converter.pymupdf_to_standard(bbox[1], 0)
                
                # Enhanced symbol detection
                symbol_type = "unknown"
                if height < 3 and width > 10:
                    symbol_type = "fraction_bar"
                elif aspect < 0.2 and height > 10:
                    symbol_type = "vertical_line"
                elif 0.8 <= aspect <= 1.2 and width > 5 and height > 5:
                    symbol_type = "square_or_circle"
                
                symbols.append({
                    "type": "vector_symbol",
                    "symbol_guess": symbol_type,
                    "bbox": [bbox[0], fixed_y0, bbox[2], fixed_y1],
                    "dimensions": {"width": width, "height": height},
                    "source": "drawing-stroke",
                    "page": page_num,
                    "extraction_method": "enhanced"
                })
                
        except Exception as e:
            print(f"    ⚠️ Enhanced vector symbol extraction failed: {e}")
            
        return symbols

    def convert_pdf_to_json_enhanced(self, pdf_path: str, output_path: str = None,
                                    include_images: bool = False,
                                    image_handling: str = "metadata") -> Dict[str, Any]:
        """Enhanced PDF to JSON conversion with consistent coordinates"""
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"🚀 ENHANCED PDF Conversion: {Path(pdf_path).name}")
        print(f"   Image handling: {image_handling}")
        print(f"   Include images: {include_images}")

        # Single PDF opening for all operations
        with pdfplumber.open(pdf_path) as pdf:
            fitz_doc = fitz.open(pdf_path)
            
            pdf_data = {
                "metadata": self._extract_enhanced_metadata(pdf_path, pdf),
                "pages": [],
                "extraction_settings": {
                    "coordinate_system": "standard_pdf_y0_at_bottom",
                    "units": "points", 
                    "text_cleaning": "enhanced_ocr_correction",
                    "image_handling": image_handling,
                    "include_images": include_images,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            for page_num, page in enumerate(pdf.pages):
                print(f"  📄 Processing page {page_num + 1}/{len(pdf.pages)}...")
                
                # Unified coordinate converter for this page
                coord_converter = CoordinateConverter(page.height)
                
                page_data = {
                    "page_number": page_num + 1,
                    "dimensions": {
                        "width": float(page.width), 
                        "height": float(page.height),
                        "rotation": page.rotation or 0
                    },
                    "text_content": self._extract_text_enhanced(page, coord_converter),
                    "tables": self._extract_tables_enhanced(page, coord_converter),
                    "images": self._extract_images_enhanced(page, include_images, image_handling, page_num, fitz_doc, coord_converter),
                    "layout_elements": self._extract_layout_enhanced(page, coord_converter),
                    "vector_symbols": self._extract_vector_symbols_enhanced(fitz_doc, page_num, coord_converter)
                }
                
                pdf_data["pages"].append(page_data)
            
            fitz_doc.close()
        
        # Apply enhanced text processing
        pdf_data = self._apply_enhanced_text_processing(pdf_data)
        
        # Comprehensive validation
        validation = self._comprehensive_validation(pdf_data)
        pdf_data["validation"] = validation
        
        # Save with better structure
        if output_path:
            self._save_enhanced_json(pdf_data, output_path)
        
        return pdf_data

    def _extract_enhanced_metadata(self, pdf_path, pdf):
        """Extract enhanced metadata"""
        path = Path(pdf_path)
        return {
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "file_format": "PDF",
            "total_pages": len(pdf.pages),
            "extraction_date": datetime.now().isoformat(),
            "pdf_version": getattr(pdf, 'metadata', {}).get('PDF', 'Unknown')
        }

    def _apply_enhanced_text_processing(self, pdf_data):
        """Apply enhanced text cleaning and merging"""
        enhanced_data = pdf_data.copy()
        
        for page in enhanced_data["pages"]:
            text_content = page.get("text_content", [])
            
            if text_content:
                # Clean and classify each block
                cleaned_blocks = []
                for block in text_content:
                    cleaned_block = block.copy()
                    original_content = block.get("content", "")
                    cleaned_content = enhanced_clean_extracted_text(original_content)
                    cleaned_block["content"] = cleaned_content
                    
                    improved_type = improved_content_classification(cleaned_content)
                    cleaned_block["content_type"] = improved_type
                    
                    # Update translation readiness
                    if improved_type in ["telugu_text", "odia_text", "mathematical", "data_label", "exam_reference"]:
                        cleaned_block["translation_ready"] = False
                    
                    cleaned_blocks.append(cleaned_block)
                
                # Merge fragmented blocks
                merged_blocks = merge_fragmented_text_blocks(cleaned_blocks)
                page["text_content"] = merged_blocks
                
                # Log improvements
                original_count = len(text_content)
                merged_count = len(merged_blocks)
                if merged_count < original_count:
                    print(f"   ✅ Merged {original_count - merged_count} fragmented text blocks")
        
        return enhanced_data

    def _comprehensive_validation(self, pdf_data):
        """Detailed validation of extracted content"""
        validation = {
            "summary": {},
            "page_breakdown": [],
            "issues": [],
            "recommendations": []
        }
        
        total_text_blocks = 0
        total_tables = 0
        total_images = 0
        total_vector_symbols = 0
        
        for page in pdf_data["pages"]:
            page_stats = {
                "page": page["page_number"],
                "text_blocks": len(page.get("text_content", [])),
                "tables": len(page.get("tables", [])),
                "images": len(page.get("images", [])),
                "vector_symbols": len(page.get("vector_symbols", [])),
                "quality_indicators": self._analyze_page_quality(page)
            }
            
            validation["page_breakdown"].append(page_stats)
            total_text_blocks += page_stats["text_blocks"]
            total_tables += page_stats["tables"]
            total_images += page_stats["images"]
            total_vector_symbols += page_stats["vector_symbols"]
        
        # Overall summary
        validation["summary"] = {
            "total_pages": len(pdf_data["pages"]),
            "total_text_blocks": total_text_blocks,
            "total_tables": total_tables,
            "total_images": total_images,
            "total_vector_symbols": total_vector_symbols,
            "extraction_completeness": self._calculate_completeness_score(pdf_data)
        }
        
        # Generate recommendations
        if total_text_blocks == 0:
            validation["recommendations"].append("Enable OCR for better text extraction")
        if total_vector_symbols > 0:
            validation["recommendations"].append("Vector symbols detected - consider mathematical content processing")
        if total_tables == 0:
            validation["recommendations"].append("No tables detected - check table extraction settings")
        
        return validation

    def _analyze_page_quality(self, page):
        """Analyze quality indicators for a page"""
        text_blocks = page.get("text_content", [])
        if not text_blocks:
            return {"score": 0, "issues": ["No text content"]}
        
        issues = []
        valid_blocks = 0
        
        for block in text_blocks:
            content = block.get("content", "")
            if len(content.strip()) > 1:  # Non-empty content
                valid_blocks += 1
            else:
                issues.append("Empty text block")
        
        score = valid_blocks / len(text_blocks) if text_blocks else 0
        return {"score": score, "issues": issues, "valid_blocks": valid_blocks}

    def _calculate_completeness_score(self, pdf_data):
        """Calculate overall extraction completeness score"""
        total_pages = len(pdf_data["pages"])
        if total_pages == 0:
            return 0
        
        page_scores = []
        for page in pdf_data["pages"]:
            quality = self._analyze_page_quality(page)
            page_scores.append(quality["score"])
        
        return sum(page_scores) / len(page_scores) if page_scores else 0

    def _save_enhanced_json(self, data: Dict[str, Any], output_path: str):
        """Save JSON with better formatting and error handling"""
        try:
            # Create directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            file_size = Path(output_path).stat().st_size
            print(f"✅ Enhanced JSON saved: {output_path} ({file_size / 1024:.1f} KB)")
            
        except Exception as e:
            print(f"❌ Failed to save enhanced JSON: {e}")
            raise

# ============================================================
# TRANSLATOR CLASS
# ============================================================
class JSONTranslator:
    def __init__(self):
        self.output_dir = "translated_jsons"
        Path(self.output_dir).mkdir(exist_ok=True)

    def translate_json_file(self, input_json: str, target_lang: str, lang_name: str):
        """Translate JSON file into given language while preserving ALL elements including coordinates."""    
        # Check if input file exists
        if not os.path.exists(input_json):
            print(f"❌ Error: Input file '{input_json}' not found!")
            return None
        
        # Load original data FIRST
        with open(input_json, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        # ⭐⭐ DEEP COPY TO PRESERVE ORIGINAL STRUCTURE ⭐⭐
        data = copy.deepcopy(original_data)  # This preserves ALL original data including coordinates
        
        # 🖼️ FIX: Preserve all image data from the original JSON
        preserved_images = 0
        for page_index, (orig_page, trans_page) in enumerate(zip(original_data.get("pages", []), data.get("pages", []))):
            orig_images = orig_page.get("images", [])
            trans_images = trans_page.get("images", [])
            for img_index, (orig_img, trans_img) in enumerate(zip(orig_images, trans_images)):
                if not trans_img.get("data") and orig_img.get("data"):
                    trans_img["data"] = orig_img["data"]
                    preserved_images += 1
                    print(f"   ✅ Preserved image {img_index+1} on page {page_index+1}")
        if preserved_images > 0:
            print(f"\n🖼️ TOTAL IMAGES PRESERVED FROM ORIGINAL JSON: {preserved_images}\n")
        else:
            print("\n⚠️ No images required preservation (all already intact)\n")

        # ---- Continue with original logic ----
        original_counts = count_json_elements(original_data)
        print(f"🔍 PRE-TRANSLATION ANALYSIS:")
        print(f"   • Vector symbols: {original_counts['vector_symbols']}")
        print(f"   • Layout lines: {original_counts['layout_elements']}")
        print(f"   • Images: {original_counts['images']}")
        print(f"   • Tables: {original_counts['tables']}")
        print(f"   • Total text blocks: {original_counts['text_blocks']}")
        print(f"   • Translatable blocks: {original_counts['translatable_blocks']}")

        total_translated = 0
        total_skipped = 0
        output_dir_path = Path(self.output_dir)
        output_dir_path.mkdir(exist_ok=True)
        print(f"\n🌐 TRANSLATING → {lang_name} ({target_lang})")
        print("=" * 80)

        # ⭐⭐ SMART TRANSLATION WITH CONTENT FILTERING ⭐⭐
        for p, page in enumerate(data.get("pages", []), start=1):
            page_translated = 0
            page_skipped = 0
            
            print(f"\n📄 Page {p}:")
            for block_idx, block in enumerate(page.get("text_content", []), start=1):
                original_text = block.get("content", "").strip()
                content_type = block.get("content_type", "")
                
                if not original_text:
                    continue

                # ⭐⭐ SMART CONTENT FILTERING ⭐⭐
                if not should_translate_content(original_text, content_type):
                    print(f"   ⏭️  SKIPPED [{block_idx}]: {original_text[:60]}...")
                    page_skipped += 1
                    total_skipped += 1
                    continue

                # ⭐⭐ TRANSLATE ONLY APPROPRIATE CONTENT ⭐⭐
                print(f"   🔄 TRANSLATING [{block_idx}]: {original_text[:60]}...")            
                translated_text = safe_translate(original_text, target_lang)
                
                # ⭐⭐ CRITICAL: Only add translated_content field, preserve all existing fields ⭐⭐
                block["translated_content"] = translated_text
                block["translation_metadata"] = {
                    "original_language": "en",
                    "target_language": target_lang,
                    "content_type": content_type,
                    "translated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                page_translated += 1
                total_translated += 1

                # Show side-by-side preview
                src_preview = original_text[:80] + "..." if len(original_text) > 80 else original_text
                trans_preview = translated_text[:80] + "..." if len(translated_text) > 80 else translated_text
                
                print(f"     🟦 English : {src_preview}")
                print(f"     🟩 {lang_name}: {trans_preview}")
                print("     " + "-" * 70)

            print(f"   📊 Page {p} summary: {page_translated} translated, {page_skipped} skipped")

        # ⭐⭐ POST-TRANSLATION VERIFICATION ⭐⭐
        final_counts = count_json_elements(data)
        
        print(f"\n✅ POST-TRANSLATION VERIFICATION:")
        print(f"   • Vector symbols: {final_counts['vector_symbols']} (preserved: {final_counts['vector_symbols'] == original_counts['vector_symbols']})")
        print(f"   • Layout lines: {final_counts['layout_elements']} (preserved: {final_counts['layout_elements'] == original_counts['layout_elements']})")
        print(f"   • Images: {final_counts['images']} (preserved: {final_counts['images'] == original_counts['images']})")
        print(f"   • Tables: {final_counts['tables']} (preserved: {final_counts['tables'] == original_counts['tables']})")
        print(f"   • Translation stats: {total_translated} translated, {total_skipped} skipped")
        
        # ⭐⭐ COORDINATE PRESERVATION CHECK ⭐⭐
        coordinate_check = verify_coordinate_preservation(original_data, data)
        
        # Check if all structural elements are preserved
        all_preserved = (
            final_counts['vector_symbols'] == original_counts['vector_symbols'] and
            final_counts['layout_elements'] == original_counts['layout_elements'] and
            final_counts['images'] == original_counts['images'] and
            final_counts['tables'] == original_counts['tables'] and
            coordinate_check
        )
        
        if all_preserved:
            print("🎉 SUCCESS: All structural elements AND coordinates preserved perfectly!")
        else:
            print("❌ WARNING: Some elements may have been modified!")

        # Save file with absolute path
        output_path = output_dir_path / f"{Path(input_json).stem}_{target_lang}.json"
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ SUCCESSFULLY SAVED TRANSLATED FILE:")
            print(f"   📍 Location: {output_path.absolute()}")
            print(f"   📊 File size: {output_path.stat().st_size} bytes")
            print(f"   🔤 Translation stats: {total_translated} translated, {total_skipped} skipped")
            print(f"   🏗️  Structural elements preserved: {all_preserved}")
            print(f"   📐 Coordinates preserved: {coordinate_check}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None

# ============================================================
# PDF GENERATOR CLASS
# ============================================================
class PDFGenerator:
    def __init__(self, json_data, output_pdf):
        if isinstance(json_data, str):
            with open(json_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = json_data

        self.output_pdf = output_pdf
        self._register_fonts()

    def _register_fonts(self):
        """Register custom fonts for ReportLab fallback"""
        try:
            for lang, font_path in FONTS.items():
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(lang.capitalize(), font_path))
                    print(f"✅ Registered font: {lang}")
        except Exception as e:
            print(f"⚠️ Could not register custom fonts: {e}")

    def clean(self, s):
        if not s:
            return ""
        s = html.unescape(str(s))
        return s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()

    def detect_language(self, data):
        """Auto-detect which language is used in the JSON"""
        sample = json.dumps(data, ensure_ascii=False).lower()
        if "telugu" in sample:
            return "telugu"
        elif "hindi" in sample:
            return "hindi" 
        elif "odia" in sample or "oriya" in sample:
            return "odia"
        else:
            return "telugu"

    def build_page_html(self, page_data, lang):
        """Build HTML for a single page using reference code approach"""
        font_file = FONTS.get(lang, FONTS["telugu"])
        font_path = pathlib.Path(font_file).resolve().as_uri() if os.path.exists(font_file) else ""

        css = f"""
        @font-face {{
            font-family: 'LangFont';
            src: url('{font_path}') format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'LangFont', sans-serif;
            background: white;
            color: black;
            position: relative;
            width: 595px;
            height: 842px;
            margin: 0;
            padding: 0;
        }}
        .text-element {{
            position: absolute;
            white-space: pre-wrap;
            background: transparent;
            color: black;
        }}
        .image-element {{
            position: absolute;
            border: none;
        }}
        .line-element {{
            position: absolute;
            background: black;
            transform-origin: 0 0;
        }}
        .rect-element {{
            position: absolute;
            border: 1px solid black;
            background: transparent;
        }}
        """

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            f"<style>{css}</style>",
            "</head>",
            "<body style='background: white; margin: 0; padding: 0;'>"
        ]

        # Add images
        for image in page_data.get("images", []):
            x0 = image['position']['x0']
            y0 = 842 - image['position']['y0']  # Convert Y coordinate
            width = image['position']['width']
            height = image['position']['height']
            image_data = image.get('data', '')
            
            if image_data:
                try:
                    img_src = f"data:image/png;base64,{image_data}"
                    html_parts.append(
                        f'<img class="image-element" src="{img_src}" '
                        f'style="left:{x0}px; top:{y0}px; width:{width}px; height:{height}px;">'
                    )
                except Exception as e:
                    print(f"⚠️ Error adding image: {e}")

        # Add text elements - use translated content only, avoid adding original content twice
        for text_block in page_data.get("text_content", []):
            # Only use translated content if available
            content = text_block.get("translated_content", "") or text_block.get("content", "")
            if not content:
                continue

            x0 = text_block['position']['x0']
            y0 = 842 - text_block['position']['y0']  # Convert Y coordinate
            font_size = text_block.get('font', {}).get('size', 12)
            
            # Clean the content
            content = self.clean(content)
            
            html_parts.append(
                f'<div class="text-element" '
                f'style="left:{x0}px; top:{y0}px; font-size:{font_size}px; background: transparent; color: black;">'
                f'{content}'
                f'</div>'
            )

        # Add layout elements (lines)
        for element in page_data.get('layout_elements', []):
            if element.get("type") == "line":
                x0 = element['position']['x0']
                y0 = 842 - element['position']['y0']  # Convert Y coordinate
                x1 = element['position']['x1']
                y1 = 842 - element['position']['y1']  # Convert Y coordinate
                
                # Calculate line properties
                width = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                angle = -1 * math.atan2(y1 - y0, x1 - x0) if x1 != x0 else -math.pi/2
                
                html_parts.append(
                    f'<div class="line-element" '
                    f'style="left:{x0}px; top:{y0}px; width:{width}px; height:1px; transform: rotate({angle}rad);">'
                    f'</div>'
                )

        # Add vector symbols (rectangles)
        for symbol in page_data.get('vector_symbols', []):
            x0, y0_orig, x1, y1_orig = symbol['bbox']
            y0 = 842 - y1_orig  # Convert Y coordinate
            height = y1_orig - y0_orig
            width = x1 - x0
            
            html_parts.append(
                f'<div class="rect-element" '
                f'style="left:{x0}px; top:{y0}px; width:{width}px; height:{height}px;">'
                f'</div>'
            )

        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def generate_pdf(self):
        """Generate PDF using Playwright approach, with fallback to ReportLab"""
        pages = self.data.get("pages", [])
        if not pages:
            print("❌ No pages found in the JSON!")
            return

        # Detect language
        lang = self.detect_language(self.data)
        print(f"🈶 Detected language: {lang}")

        try:
            # Attempt to generate PDF using Playwright
            pdf_merger = PdfMerger()
            temp_pdfs = []

            # Use Playwright to generate PDF for each page
            with sync_playwright() as p:
                browser = p.chromium.launch()

                for page_num, page_data in enumerate(pages):
                    print(f"📄 Processing page {page_num + 1}/{len(pages)}")

                    # Build HTML for this page
                    html_content = self.build_page_html(page_data, lang)

                    # Create temporary HTML file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                        f.write(html_content)
                        html_path = f.name

                    # Create new page for each page
                    page = browser.new_page()
                    page.set_viewport_size({"width": 595, "height": 842})
                    page.goto(f"file://{html_path}")

                    page.wait_for_timeout(1000)  # Wait for the content to load

                    # Generate PDF for this page
                    temp_pdf = f"temp_page_{page_num}.pdf"
                    page.pdf(
                        path=temp_pdf,
                        width="595px",
                        height="842px", 
                        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                        print_background=True
                    )

                    pdf_merger.append(temp_pdf)
                    temp_pdfs.append(temp_pdf)

                    page.close()
                    os.unlink(html_path)

                browser.close()

                # Merge all pages into final PDF
                pdf_merger.write(self.output_pdf)
                pdf_merger.close()

                # Clean up temporary PDFs
                for temp_pdf in temp_pdfs:
                    if os.path.exists(temp_pdf):
                        os.unlink(temp_pdf)

                print(f"✅ PDF created successfully with {len(pages)} pages: {self.output_pdf}")
        
        except Exception as e:
            print(f"❌ Error creating PDF with Playwright: {e}")
            print("🔄 Falling back to ReportLab approach...")

            # If Playwright fails, fallback to ReportLab approach
            self.generate_pdf_fallback()

    def generate_pdf_fallback(self):
        """Fallback to ReportLab approach"""
        print("🔄 Using ReportLab fallback...")
        pages = self.data.get("pages", [])
        pdf = canvas.Canvas(self.output_pdf, pagesize=(595, 842))

        for page_num, page_data in enumerate(pages):
            if page_num > 0:
                pdf.showPage()

            self.draw_images(pdf, page_data)
            self.draw_text(pdf, page_data)
            self.draw_layout_elements(pdf, page_data)
            self.draw_vector_symbols(pdf, page_data)

        pdf.save()
        print(f"✅ PDF created (fallback): {self.output_pdf}")

    def draw_images(self, pdf, page_data):
        """Draw images on PDF page"""
        images = page_data.get("images", [])
        for image in images:
            x0 = image['position']['x0']
            y0 = image['position']['y0']
            width = image['position']['width']
            height = image['position']['height']

            image_data = image.get('data', '')
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    img_reader = ImageReader(BytesIO(image_bytes))
                    pdf.drawImage(img_reader, x0, y0, width, height)
                except Exception as e:
                    print(f"⚠️ Error rendering image: {e}")

    def draw_text(self, pdf, page_data):
        """Draw text on PDF page"""
        text_blocks = page_data.get("text_content", [])
        for block in text_blocks:
            translated_content = block.get("translated_content", "")
            original_content = block.get("content", "")
            
            if translated_content and translated_content.strip():
                content = translated_content
                print(f"✅ Using TRANSLATED: {content[:30]}...")
            else:
                content = original_content
                print(f"⚠️ No translation, using ORIGINAL: {content[:30]}...")

            if not content:
                continue

            x0 = block['position']['x0']
            y0 = block['position']['y0']
            font_size = block.get('font', {}).get('size', 12)

            # Try to use custom font, fallback to Helvetica
            try:
                lang = self.detect_language(self.data)
                font_name = lang.capitalize()
                pdf.setFont(font_name, font_size)
            except:
                pdf.setFont("Helvetica", font_size)

            pdf.drawString(x0, y0, self.clean(content))

    def draw_layout_elements(self, pdf, page_data):
        """Draw layout elements (lines) on PDF page"""
        layout_elements = page_data.get('layout_elements', [])
        for element in layout_elements:
            if element.get("type") == "line":
                x0 = element['position']['x0']
                y0 = element['position']['y0']
                x1 = element['position']['x1']
                y1 = element['position']['y1']
                pdf.setStrokeColor(colors.black)
                pdf.setLineWidth(0.5)
                pdf.line(x0, y0, x1, y1)

    def draw_vector_symbols(self, pdf, page_data):
        """Draw vector symbols on PDF page"""
        vector_symbols = page_data.get('vector_symbols', [])
        for symbol in vector_symbols:
            x0, y0, x1, y1 = symbol['bbox']
            pdf.setStrokeColor(colors.black)
            pdf.setLineWidth(1.0)
            pdf.rect(x0, y0, x1 - x0, y1 - y0, stroke=1, fill=0)

# ============================================================
# MAIN PIPELINE CLASS
# ============================================================
class PDFProcessingPipeline:
    def __init__(self):
        self.pdf_converter = PDFToJSONConverter()
        self.translator = JSONTranslator()
        
    def run_complete_pipeline(self, pdf_path: str):
        """Run complete pipeline: PDF → JSON → Translation → PDF"""
        print("=" * 80)
        print("🚀 COMPLETE PDF PROCESSING PIPELINE")
        print("=" * 80)
        
        # Step 1: PDF to JSON Extraction
        print("\n📄 STEP 1: PDF TO JSON EXTRACTION")
        print("-" * 40)
        
        json_output = "extracted_pdf.json"
        try:
            extracted_data = self.pdf_converter.convert_pdf_to_json_enhanced(
                pdf_path, 
                json_output,
                include_images=True,
                image_handling="metadata"
            )
            print(f"✅ PDF extraction completed: {json_output}")
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            return
        
        # Step 2: Translation
        print("\n🌐 STEP 2: TRANSLATION")
        print("-" * 40)
        
        print("🎯 SELECT TRANSLATION LANGUAGES:")
        for key, (code, name) in LANG_OPTIONS.items():
            print(f"  {key}. {name}")
        
        selection = input("\nEnter your choices (e.g. 1,2,3): ").strip()
        if not selection:
            print("❌ No languages selected. Exiting.")
            return

        choices = [s.strip() for s in selection.split(",") if s.strip() in LANG_OPTIONS]
        
        if not choices:
            print("❌ Invalid selection. Please choose valid numbers from the list.")
            return

        translated_files = []
        for choice in choices:
            code, name = LANG_OPTIONS[choice]
            print(f"\n{'='*80}")
            print(f"PROCESSING: {name} ({code})")
            print(f"{'='*80}")
            
            result = self.translator.translate_json_file(json_output, code, name)
            if result:
                translated_files.append((result, name, code))
            
            # Small delay between languages to avoid API rate limits
            if len(choices) > 1:
                time.sleep(2)
        
        # Step 3: PDF Generation
        print("\n📄 STEP 3: PDF GENERATION")
        print("-" * 40)
        
        for translated_file, lang_name, lang_code in translated_files:
            print(f"\n🔄 Generating PDF for {lang_name}...")
            output_pdf = f"output_{lang_name}.pdf"
            
            try:
                pdf_gen = PDFGenerator(str(translated_file), output_pdf)
                pdf_gen.generate_pdf()
                print(f"✅ PDF generated: {output_pdf}")
            except Exception as e:
                print(f"❌ PDF generation failed for {lang_name}: {e}")
        
        # Final Summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Processing Summary:")
        print(f"   • Input PDF: {pdf_path}")
        print(f"   • Extracted JSON: {json_output}")
        print(f"   • Languages processed: {len(translated_files)}")
        for file, lang_name, _ in translated_files:
            print(f"   • {lang_name}: {file.name}")
        print(f"   • Output PDFs: Generated in current directory")
        print(f"\n💡 All files are ready for use!")

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    pipeline = PDFProcessingPipeline()
    
    print("🎯 PDF PROCESSING PIPELINE - COMPLETE WORKFLOW")
    print("=" * 60)
    
    # Get PDF file path
    pdf_path = input("Enter PDF file path (or press Enter for default): ").strip()
    if not pdf_path:
        pdf_path = "SBI Clerk Prelims.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please make sure the file exists in the current directory.")
        return
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(pdf_path)

if __name__ == "__main__":
    main()