"""
Professional Interactive Translator for JSON Documents
------------------------------------------------------
→ Lets you select Indian languages (Hindi, Telugu, Odia, etc.)
→ Shows live translation in console  
→ Saves layout-faithful translated JSON
→ VERIFIES vector symbols and layout lines are preserved
→ PRESERVES COORDINATES during translation
→ SMART CONTENT FILTERING for better translation quality
→ SUPPORTS SEPARATED MATH/NUMERIC CONTENT
"""

import json
import time
import os
import copy
import re
from deep_translator import GoogleTranslator
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_JSON = "sbi_extracted.json"  # your extracted JSON file
OUTPUT_DIR = "translated_jsons"
MAX_RETRIES = 3
RETRY_DELAY = 2

# ============================================================
# LANGUAGE OPTIONS  
# ============================================================
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

# ============================================================
# ENHANCED CONTENT FILTERING - CRITICAL FOR GOVERNMENT EXAMS
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
    if content_type and content_type in ['mathematical', 'data_label', 'option', 'coordinate']:
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
    
    # FIXED: IMPROVED TEXT+NUMBER DETECTION
    # Check if this is descriptive text with numbers (should be translated)
    words = re.findall(r'\b[a-zA-Z]+\b', text_clean)
    numbers = re.findall(r'\b\d+\b', text_clean)
    
    # If it has both words AND numbers, and at least 2 words, it's descriptive text
    if len(words) >= 2 and len(numbers) >= 1:
        return True
    
    # If it's mostly text with some symbols, translate it
    if len(words) >= 3:
        return True
    
    return True

def get_content_category(text: str) -> str:
    """Categorize content for better translation handling"""
    if not text:
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # Instructions
    if any(word in text_lower for word in ['directions', 'study the', 'answer the', 'following questions', 'read the']):
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
    
    # Mathematical content
    elif any(symbol in text for symbol in ['÷', '×', '=', '%', '√', '²', '³']):
        return "mathematical"
    
    # Headers
    elif re.match(r'^[A-Z][A-Z\s]+$', text.strip()):
        return "header"
    
    else:
        return "normal_text"

# ============================================================
# HELPERS - ENHANCED WITH COORDINATE PRESERVATION
# ============================================================
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
    math_blocks = 0
    translatable_blocks = 0
    
    for page in data.get("pages", []):
        vector_count += len(page.get('vector_symbols', []))
        layout_count += len(page.get('layout_elements', []))
        image_count += len(page.get('images', []))
        table_count += len(page.get('tables', []))
        math_blocks += len(page.get('math_numeric_content', []))
        
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
        'math_blocks': math_blocks,
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
        # Check vector symbols coordinates
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
        
        # Check layout elements coordinates
        orig_layouts = orig_page.get("layout_elements", [])
        trans_layouts = trans_page.get("layout_elements", [])
        
        if len(orig_layouts) != len(trans_layouts):
            print(f"❌ Page {page_idx + 1}: Layout element count mismatch")
            all_coordinates_preserved = False
        else:
            for layout_idx, (orig_layout, trans_layout) in enumerate(zip(orig_layouts, trans_layouts)):
                # FIXED: Compare bbox instead of position
                orig_bbox = orig_layout.get("bbox", [])
                trans_bbox = trans_layout.get("bbox", [])
                if orig_bbox != trans_bbox:
                    print(f"❌ Page {page_idx + 1}, Layout {layout_idx + 1}: BBOX coordinates changed")
                    all_coordinates_preserved = False
        
        # Check text content coordinates
        orig_text_blocks = orig_page.get("text_content", [])
        trans_text_blocks = trans_page.get("text_content", [])
        
        if len(orig_text_blocks) != len(trans_text_blocks):
            print(f"❌ Page {page_idx + 1}: Text block count mismatch")
            all_coordinates_preserved = False
            continue
        
        for block_idx, (orig_block, trans_block) in enumerate(zip(orig_text_blocks, trans_text_blocks)):
            # FIXED: Compare bbox instead of position
            orig_bbox = orig_block.get("bbox", [])
            trans_bbox = trans_block.get("bbox", [])
            
            # Compare all bbox coordinates
            if orig_bbox != trans_bbox:
                print(f"❌ Page {page_idx + 1}, Block {block_idx + 1}: BBOX coordinates changed!")
                print(f"   Original: {orig_bbox}")
                print(f"   Translated: {trans_bbox}")
                all_coordinates_preserved = False
        
        # Check math_numeric_content coordinates
        orig_math_blocks = orig_page.get("math_numeric_content", [])
        trans_math_blocks = trans_page.get("math_numeric_content", [])
        
        if len(orig_math_blocks) != len(trans_math_blocks):
            print(f"❌ Page {page_idx + 1}: Math/numeric block count mismatch")
            all_coordinates_preserved = False
        else:
            for math_idx, (orig_math, trans_math) in enumerate(zip(orig_math_blocks, trans_math_blocks)):
                # FIXED: Compare bbox instead of position
                orig_bbox = orig_math.get("bbox", [])
                trans_bbox = trans_math.get("bbox", [])
                
                if orig_bbox != trans_bbox:
                    print(f"❌ Page {page_idx + 1}, Math block {math_idx + 1}: BBOX coordinates changed!")
                    print(f"   Original: {orig_bbox}")
                    print(f"   Translated: {trans_bbox}")
                    all_coordinates_preserved = False
    
    if all_coordinates_preserved:
        print("✅ SUCCESS: All coordinates preserved during translation!")
    else:
        print("❌ WARNING: Some coordinates were modified during translation!")
    
    return all_coordinates_preserved

def check_text_number_combination(json_data):
    """Check how text+number combinations are handled"""
    print("\n🔍 CHECKING TEXT + NUMBER COMBINATIONS")
    print("=" * 50)
    
    text_with_numbers = 0
    pure_text = 0
    pure_numbers = 0
    mixed_content_blocks = []
    
    for page_num, page in enumerate(json_data.get('pages', [])):
        for block in page.get('text_content', []):
            content = block.get('content', '')
            
            # Count words and numbers
            words = re.findall(r'\b[a-zA-Z]+\b', content)
            numbers = re.findall(r'\b\d+\b', content)
            
            if words and numbers:
                text_with_numbers += 1
                if text_with_numbers <= 5:  # Show first 5 examples
                    mixed_content_blocks.append(content)
                    print(f"   📝 Text+Numbers [{len(words)}w, {len(numbers)}n]: '{content[:80]}...'")
            elif words:
                pure_text += 1
            elif numbers:
                pure_numbers += 1
    
    print(f"\n📊 CLASSIFICATION SUMMARY:")
    print(f"   Text + Numbers blocks: {text_with_numbers}")
    print(f"   Pure Text blocks: {pure_text}") 
    print(f"   Pure Numbers blocks: {pure_numbers}")
    
    return text_with_numbers, mixed_content_blocks

def check_translated_json_coordinates():
    """Check if existing translated JSON preserved coordinates"""
    print("\n🔍 CHECKING EXISTING TRANSLATED JSON COORDINATES")
    print("=" * 50)
    
    original_json = "sbi_extracted.json"
    translated_json = "translated_jsons/sbi_extracted_te.json"
    
    if not os.path.exists(original_json) or not os.path.exists(translated_json):
        print("❌ One or both JSON files not found for coordinate check")
        return False
    
    try:
        with open(original_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        with open(translated_json, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)
        return verify_coordinate_preservation(original_data, translated_data)
    except Exception as e:
        print(f"❌ Error checking translated JSON: {e}")
        return False

def translate_json_file(input_json: str, target_lang: str, lang_name: str):
    """Translate JSON file into given language while preserving ALL elements including coordinates."""    
    # Check if input file exists
    if not os.path.exists(input_json):
        print(f"❌ Error: Input file '{input_json}' not found!")
        return None
    
    # Load original data FIRST
    with open(input_json, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # ⭐⭐ NEW: Check text+number combinations before translation
    text_with_numbers_count, mixed_examples = check_text_number_combination(original_data)
    print(f"\n🔍 FOUND {text_with_numbers_count} TEXT+NUMBER BLOCKS THAT WILL BE TRANSLATED")

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
    print(f"   • Math/numeric blocks: {original_counts['math_blocks']}")
    print(f"   • Total text blocks: {original_counts['text_blocks']}")
    print(f"   • Translatable blocks: {original_counts['translatable_blocks']}")

    total_translated = 0
    total_skipped = 0
    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(exist_ok=True)
    print(f"\n🌐 TRANSLATING → {lang_name} ({target_lang})")
    print("=" * 80)

    # ⭐⭐ SMART TRANSLATION WITH CONTENT FILTERING ⭐⭐
    for p, page in enumerate(data.get("pages", []), start=1):
        page_translated = 0
        page_skipped = 0
        
        print(f"\n📄 Page {p}:")
        
        # ⭐⭐ NEW: Process math_numeric_content blocks (preserve them without translation)
        math_blocks = page.get("math_numeric_content", [])
        if math_blocks:
            print(f"   🔢 Math/Numeric blocks preserved: {len(math_blocks)}")
            for math_block in math_blocks:
                # Mark math blocks as non-translatable
                math_block["translation_ready"] = False
                math_block["content_type"] = "math_numeric"
        
        # ⭐⭐ Process text_content blocks (translate only appropriate content)
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

        print(f"   📊 Page {p} summary: {page_translated} translated, {page_skipped} skipped, {len(math_blocks)} math blocks preserved")

    # ⭐⭐ POST-TRANSLATION VERIFICATION ⭐⭐
    final_counts = count_json_elements(data)
    
    print(f"\n✅ POST-TRANSLATION VERIFICATION:")
    print(f"   • Vector symbols: {final_counts['vector_symbols']} (preserved: {final_counts['vector_symbols'] == original_counts['vector_symbols']})")
    print(f"   • Layout lines: {final_counts['layout_elements']} (preserved: {final_counts['layout_elements'] == original_counts['layout_elements']})")
    print(f"   • Images: {final_counts['images']} (preserved: {final_counts['images'] == original_counts['images']})")
    print(f"   • Tables: {final_counts['tables']} (preserved: {final_counts['tables'] == original_counts['tables']})")
    print(f"   • Math/numeric blocks: {final_counts['math_blocks']} (preserved: {final_counts['math_blocks'] == original_counts['math_blocks']})")
    print(f"   • Translation stats: {total_translated} translated, {total_skipped} skipped")
    
    # ⭐⭐ COORDINATE PRESERVATION CHECK ⭐⭐
    coordinate_check = verify_coordinate_preservation(original_data, data)
    
    # Check if all structural elements are preserved
    all_preserved = (
        final_counts['vector_symbols'] == original_counts['vector_symbols'] and
        final_counts['layout_elements'] == original_counts['layout_elements'] and
        final_counts['images'] == original_counts['images'] and
        final_counts['tables'] == original_counts['tables'] and
        final_counts['math_blocks'] == original_counts['math_blocks'] and
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
# MAIN DRIVER - ENHANCED WITH SMART FILTERING
# ============================================================
def main():
    print("\n" + "="*60)
    print("🌐 PROFESSIONAL JSON TRANSLATOR WITH SMART CONTENT FILTERING")
    print("="*60)
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📄 Input file: {INPUT_JSON}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # First check existing translated JSON if it exists
    existing_translation_check = check_translated_json_coordinates()
    
    # Check if input file exists
    if not os.path.exists(INPUT_JSON):
        print(f"\n❌ ERROR: Input file '{INPUT_JSON}' not found!")
        print("Please make sure:")
        print("1. The JSON file exists in the same directory as this script")
        print("2. The filename is spelled correctly")
        print(f"\nFiles in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"   📄 {file}")
            else:
                print(f"   📁 {file}")
        return

    print(f"\n✅ Input file found: {INPUT_JSON}")
    print(f"\n🎯 SELECT TRANSLATION LANGUAGES:")
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

    saved_files = []
    for choice in choices:
        code, name = LANG_OPTIONS[choice]
        print(f"\n{'='*80}")
        print(f"PROCESSING: {name} ({code})")
        print(f"{'='*80}")
        
        result = translate_json_file(INPUT_JSON, code, name)
        if result:
            saved_files.append(result)
        
        # Small delay between languages to avoid API rate limits
        if len(choices) > 1:
            time.sleep(2)

    print("\n" + "="*80)
    print("🎉 TRANSLATION SUMMARY:")
    print(f"📊 Total languages processed: {len(choices)}")
    print(f"💾 Files successfully saved: {len(saved_files)}")
    
    for file_path in saved_files:
        file_size = file_path.stat().st_size
        print(f"   ✅ {file_path.name} ({file_size} bytes)")
    
    print(f"\n📁 All translated files are saved in: {Path(OUTPUT_DIR).absolute()}")
    
    # Final verification
    print(f"\n🔍 FINAL VERIFICATION:")
    final_check = check_translated_json_coordinates()
    
    if final_check:
        print("🎉 PERFECT! Your translated JSON files are ready for PDF generation!")
    else:
        print("⚠️  Some coordinate issues detected. Consider re-running translation.")
    
    # Final reminder
    print(f"\n💡 IMPORTANT FEATURES:")
    print(f"   • SMART FILTERING: Mathematical content, options, and data labels are preserved")
    print(f"   • SEPARATED MATH/NUMBERS: Math symbols and numbers are in separate blocks")
    print(f"   • COORDINATE PRESERVATION: All layout coordinates are maintained")
    print(f"   • STRUCTURAL INTEGRITY: Vector symbols, layout lines, images, math blocks preserved")
    print(f"   • TRANSLATION METADATA: Added translation tracking information")
    print(f"   • READY FOR PDF GENERATION: All elements preserved for accurate reconstruction!")

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    main()