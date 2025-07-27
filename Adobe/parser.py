import fitz  # PyMuPDF
import os
import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

ROUND_DIGIT = 2
MIN_HEADING_LENGTH = 3
MAX_HEADING_LENGTH = 200
HEADING_SIZE_THRESHOLD = 1.15  # Must be 15% larger than average
HEADER_FOOTER_MARGIN = 50  # Reduced from 100 for better detection
MAX_LINE_GAP = 2.5  # Maximum gap between continuation lines

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    return re.sub(r'\s+', ' ', text.strip())

def is_likely_heading(text: str, size: float, avg_size: float, font_name: str = "") -> bool:
    """Enhanced heading detection with font analysis"""
    text = clean_text(text)
    
    # Basic length filters
    if len(text) < MIN_HEADING_LENGTH or len(text) > MAX_HEADING_LENGTH:
        return False
    
    # Size must be significantly larger than average
    if size < avg_size * HEADING_SIZE_THRESHOLD:
        return False
    
    # Filter out obvious non-headings
    non_heading_patterns = [
        r'^\d+$',  # Just numbers
        r'^[^\w\s]+$',  # Just punctuation
        r'^\d+\.\d+$',  # Decimal numbers
        r'^[ivxlcdm]+\.?$',  # Roman numerals
        r'^\([^)]*\)$',  # Text in parentheses only
    ]
    
    for pattern in non_heading_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False
    
    # Filter out common non-heading words
    if text.lower() in ['page', 'table', 'figure', 'chart', 'appendix', 
                       'bibliography', 'references', 'index', 'contents']:
        return False
    
    # Positive indicators for headings
    heading_indicators = 0
    
    # Font-based indicators
    if 'bold' in font_name.lower():
        heading_indicators += 1
    
    # Text structure indicators
    if text[0].isupper():  # Starts with capital
        heading_indicators += 1
    
    if text.isupper() and len(text) > 6:  # All caps
        heading_indicators += 2
    
    # Title case detection
    words = text.split()
    if len(words) > 1 and sum(1 for word in words if word[0].isupper()) >= len(words) * 0.7:
        heading_indicators += 1
    
    # Sentence structure (headings often don't end with periods unless they're single words)
    if not text.endswith('.') or len(words) == 1:
        heading_indicators += 1
    
    # Check for numbered headings
    if re.match(r'^\d+\.?\s+[A-Z]', text):
        heading_indicators += 2
    
    return heading_indicators >= 2

def calculate_font_statistics(doc: fitz.Document) -> Tuple[float, List[float], Dict[float, int]]:
    """Calculate comprehensive font statistics"""
    font_sizes = []
    font_size_counts = Counter()
    total_weighted_size = 0
    total_chars = 0
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        
        for block in blocks:
            # Skip potential headers/footers
            bbox = block.get('bbox', [0, 0, 0, 0])
            if bbox[1] < HEADER_FOOTER_MARGIN or bbox[3] > page_height - HEADER_FOOTER_MARGIN:
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                        
                    size = round(span["size"], ROUND_DIGIT)
                    text_length = len(text)
                    
                    font_sizes.append(size)
                    font_size_counts[size] += text_length
                    total_weighted_size += size * text_length
                    total_chars += text_length
    
    avg_font_size = total_weighted_size / total_chars if total_chars > 0 else 12
    unique_sizes = sorted(set(font_sizes), reverse=True)
    
    return avg_font_size, unique_sizes, dict(font_size_counts)

def extract_page_headings(page: fitz.Page, heading_sizes: List[float], 
                         avg_font_size: float, page_num: int) -> List[Dict]:
    """Extract headings from a single page with improved logic"""
    blocks = page.get_text("dict")["blocks"]
    page_height = page.rect.height
    heading_candidates = []
    
    for block in blocks:
        bbox = block.get('bbox', [0, 0, 0, 0])
        # Skip headers and footers
        if bbox[1] < HEADER_FOOTER_MARGIN or bbox[3] > page_height - HEADER_FOOTER_MARGIN:
            continue
            
        for line in block.get("lines", []):
            line_text_parts = []
            line_size = 0
            line_bbox = line.get('bbox', [0, 0, 0, 0])
            font_name = ""
            
            # Combine spans in the same line
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                    
                size = round(span["size"], ROUND_DIGIT)
                
                if size >= avg_font_size * 0.8:  # Don't ignore slightly smaller text
                    line_text_parts.append(text)
                    line_size = max(line_size, size)
                    if not font_name:
                        font_name = span.get("font", "")
            
            if not line_text_parts or line_size not in heading_sizes:
                continue
                
            line_text = " ".join(line_text_parts)
            
            if is_likely_heading(line_text, line_size, avg_font_size, font_name):
                heading_candidates.append({
                    'text': line_text,
                    'size': line_size,
                    'bbox': line_bbox,
                    'y_pos': line_bbox[1],
                    'font': font_name,
                    'page': page_num
                })
    
    return heading_candidates

def merge_multiline_headings(candidates: List[Dict]) -> List[Dict]:
    """Merge heading candidates that span multiple lines"""
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: x['y_pos'])
    merged_headings = []
    i = 0
    
    while i < len(candidates):
        current = candidates[i]
        combined_text = current['text']
        
        # Look for continuation lines
        j = i + 1
        while j < len(candidates):
            next_candidate = candidates[j]
            
            # Check if it's a continuation
            if (next_candidate['size'] == current['size'] and
                abs(next_candidate['y_pos'] - current['bbox'][3]) < 
                (current['bbox'][3] - current['bbox'][1]) * MAX_LINE_GAP):
                
                combined_text += " " + next_candidate['text']
                current['bbox'] = [
                    min(current['bbox'][0], next_candidate['bbox'][0]),
                    current['bbox'][1],
                    max(current['bbox'][2], next_candidate['bbox'][2]),
                    next_candidate['bbox'][3]
                ]
                candidates.pop(j)
            else:
                break
        
        current['text'] = clean_text(combined_text)
        merged_headings.append(current)
        i += 1
    
    return merged_headings

def determine_heading_levels(headings: List[Dict], heading_sizes: List[float]) -> List[Dict]:
    """Assign heading levels based on font size hierarchy"""
    for heading in headings:
        size_index = heading_sizes.index(heading['size'])
        
        # Map size index to heading level
        if size_index == 0:
            heading['level'] = "H1"
        elif size_index == 1:
            heading['level'] = "H2"
        elif size_index == 2:
            heading['level'] = "H3"
        elif size_index == 3:
            heading['level'] = "H4"
        else:
            heading['level'] = "H5"
    
    return headings

def extract_headings(pdf_path: str) -> Dict:
    """Main function to extract headings from PDF"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise Exception(f"Failed to open PDF: {str(e)}")
    
    if doc.page_count == 0:
        doc.close()
        return {"title": "", "outline": []}
    
    # Calculate font statistics
    avg_font_size, unique_sizes, size_counts = calculate_font_statistics(doc)
    print(f"Average font size: {avg_font_size:.2f}")
    
    # Determine heading sizes (significantly larger than average)
    heading_sizes = [size for size in unique_sizes 
                    if size >= avg_font_size * HEADING_SIZE_THRESHOLD]
    
    if not heading_sizes:
        print("Warning: No potential heading sizes found")
        doc.close()
        return {"title": "", "outline": []}
    
    print(f"Potential heading sizes: {heading_sizes}")
    
    # Extract headings from all pages
    all_headings = []
    title = ""
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_headings = extract_page_headings(page, heading_sizes, avg_font_size, page_num + 1)
        page_headings = merge_multiline_headings(page_headings)
        
        # Extract title from first page if not found
        if page_num == 0 and not title and page_headings:
            # Find the largest heading on first page as title
            largest_heading = max(page_headings, key=lambda x: x['size'])
            if largest_heading['size'] == heading_sizes[0]:
                title = largest_heading['text']
                page_headings.remove(largest_heading)
        
        all_headings.extend(page_headings)
    
    # Assign heading levels
    all_headings = determine_heading_levels(all_headings, heading_sizes)
    
    # Sort by page and position
    all_headings.sort(key=lambda x: (x['page'], x['y_pos']))
    
    # Clean up the results
    outline = []
    for heading in all_headings:
        outline.append({
            "level": heading['level'],
            "text": heading['text'],
            "page": heading['page']
        })
    
    doc.close()
    
    return {
        "title": title,
        "outline": outline,
        "stats": {
            "avg_font_size": round(avg_font_size, 2),
            "heading_sizes": heading_sizes,
            "total_headings": len(outline)
        }
    }

def process_pdf_file(input_dir: str, output_dir: str) -> None:
    """Process all PDF files in the input directory with better error handling"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    try:
        pdf_files = [f for f in os.listdir(input_dir) 
                    if f.lower().endswith('.pdf')]
    except PermissionError:
        print(f"Error: Permission denied accessing '{input_dir}'")
        return
    
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process...")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory '{output_dir}'")
        return
    
    successful = 0
    failed = 0
    
    for filename in pdf_files:
        print(f"\nProcessing '{filename}'...")
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Extract headings
            data = extract_headings(input_path)
            
            # Save results
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Successfully extracted {data['stats']['total_headings']} headings")
            if data['title']:
                print(f"  Title: {data['title'][:50]}{'...' if len(data['title']) > 50 else ''}")
            
            successful += 1
            
        except Exception as e:
            print(f"✗ Error processing '{filename}': {str(e)}")
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    input_dir = "C:/Users/abhiv/Documents/Adobe/pdfs"
    output_dir = "C:/Users/abhiv/Documents/Adobe/output_mine"
    
    process_pdf_file(input_dir, output_dir)