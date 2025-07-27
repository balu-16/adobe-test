import fitz  # PyMuPDF
import os
import json
import re
import time
from collections import Counter

ROUND_DIGIT = 2

def clean_repeated_text(text):
    """Advanced cleaning of repeated text patterns"""
    text = text.strip()
    
    # Handle specific patterns like "RFP: Reequest fquest fquest foo"
    # Remove repeated fragments within words
    text = re.sub(r'(\w{2,})\1+', r'\1', text)
    
    # Remove repeated word fragments
    text = re.sub(r'\b(\w+)\s+\1+\b', r'\1', text)
    
    # Remove repeated sequences of words
    words = text.split()
    if len(words) > 1:
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            # Look ahead to see if this word repeats
            j = i + 1
            while j < len(words) and words[j] == current_word:
                j += 1
            cleaned_words.append(current_word)
            i = j
        text = ' '.join(cleaned_words)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_likely_heading(text, size, avg_size, max_size):
    """Enhanced heading detection with better criteria"""
    text = text.strip()
    
    # Basic filters
    if len(text) < 2 or len(text) > 300:
        return False
    
    # Size should be larger than average (more lenient threshold)
    size_ratio = size / avg_size
    if size_ratio < 1.05:  # More lenient than before
        return False
    
    # Filter out common non-heading patterns
    if re.match(r'^\d+$', text):  # Just numbers
        return False
    if re.match(r'^[^\w\s]*$', text):  # Just punctuation
        return False
    if text.lower() in ['page', 'table', 'figure', 'chart', 'continued']:
        return False
    
    # Filter out URLs, emails, etc.
    if re.search(r'[@.]com|www\.|http|\.ca|\.org', text.lower()):
        return False
    
    # Very short text needs to be significantly larger
    if len(text) < 10 and size_ratio < 1.2:
        return False
    
    # Good heading indicators
    word_count = len(text.split())
    
    # Single word headings need to be quite large
    if word_count == 1 and size_ratio < 1.3:
        return False
    
    # Check for heading-like characteristics
    has_title_case = bool(re.search(r'^[A-Z][a-z]', text))
    has_caps = bool(re.search(r'[A-Z]', text))
    is_all_caps = text.isupper() and len(text) > 3
    
    # At least one heading indicator should be present
    if not (has_title_case or is_all_caps or has_caps):
        return False
    
    return True

def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    result_outline = []
    title = ""
    
    # First pass: collect font size statistics
    font_sizes = {}
    total_text_length = 0
    total_weighted_size = 0
    
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            # Skip headers and footers
            x0, y0, x1, y1 = block['bbox']
            page_height = page.rect.height
            if y1 < 80 or y0 > page_height - 80:  # Adjusted margins
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], ROUND_DIGIT)
                    
                    if text and len(text) > 0:
                        text_length = len(text)
                        if font_sizes.get(size):
                            font_sizes[size] += text_length
                        else:
                            font_sizes[size] = text_length
                        
                        total_weighted_size += size * text_length
                        total_text_length += text_length
    
    # Calculate statistics
    if total_text_length == 0:
        doc.close()
        return {"title": "", "outline": []}
    
    sizes_to_remove = []
    for size in font_sizes:
        if font_sizes[size] < 15:
            sizes_to_remove.append(size)

    for key in sizes_to_remove:
        font_sizes.pop(key)
    
    avg_font_size = total_weighted_size / total_text_length
    max_font_size = max(font_sizes) if font_sizes else avg_font_size
    
    # Find the most common font size (likely body text)
    body_font_size = max(font_sizes.keys(),key=lambda x: font_sizes[x])
    
    print(f"Average font size: {avg_font_size:.2f}")
    print(f"Most common font size (body): {body_font_size}")
    print(f"Max font size: {max_font_size}")
    
    # Determine heading font sizes (more flexible approach)
    unique_sizes = sorted(list(set(font_sizes)), reverse=True)
    heading_sizes = []
    
    for size in unique_sizes:
        # Include sizes that are notably larger than body text
        if size > body_font_size * 1.1 or size > avg_font_size * 1.05:
            heading_sizes.append(size)
    
    print(f"Heading sizes: {heading_sizes}")
    
    # Second pass: extract headings
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_headings = []
        
        for block in blocks:
            # Skip headers and footers
            x0, y0, x1, y1 = block['bbox']
            page_height = page.rect.height
            if y1 < 80 or y0 > page_height - 80:
                continue
                
            for line in block.get("lines", []):
                line_text = ""
                line_size = 0
                line_bbox = line['bbox']
                
                # Combine spans in the same line
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], ROUND_DIGIT)
                    
                    if text:
                        line_text += text + " "
                        line_size = max(line_size, size)
                
                line_text = line_text.strip()
                
                # Check if this line could be a heading
                if line_text and line_size in heading_sizes:
                    # Clean the text
                    cleaned_text = clean_repeated_text(line_text)
                    
                    if is_likely_heading(cleaned_text, line_size, avg_font_size, max_font_size):
                        page_headings.append({
                            'text': cleaned_text,
                            'size': line_size,
                            'bbox': line_bbox,
                            'y_pos': line_bbox[1]
                        })
        
        # Group nearby headings of the same size (multi-line headings)
        grouped_headings = []
        page_headings.sort(key=lambda x: x['y_pos'])
        
        i = 0
        while i < len(page_headings):
            current = page_headings[i]
            combined_text = current['text']
            current_size = current['size']
            
            # Look for continuation lines
            j = i + 1
            while j < len(page_headings):
                next_heading = page_headings[j]
                
                # Check if next line is a continuation
                y_distance = next_heading['y_pos'] - current['bbox'][3]
                estimated_line_height = current['bbox'][3] - current['bbox'][1]
                
                if (next_heading['size'] == current_size and 
                    y_distance <= estimated_line_height * 2.5):
                    
                    combined_text += " " + next_heading['text']
                    current = next_heading
                    page_headings.pop(j)
                else:
                    break
            
            # Final cleaning and validation
            combined_text = clean_repeated_text(combined_text)
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            
            if (combined_text and 
                is_likely_heading(combined_text, current_size, avg_font_size, max_font_size) and
                len(combined_text) >= 3):
                
                grouped_headings.append({
                    'text': combined_text,
                    'size': current_size,
                    'y_pos': page_headings[i]['y_pos'] if i < len(page_headings) else current['y_pos']
                })
            
            i += 1
        
        # Sort by position and process
        grouped_headings.sort(key=lambda x: x['y_pos'])
        
        # Extract actual headings
        for heading in grouped_headings:
            text = heading['text']
            size = heading['size']
            
            # Determine if this is the title (first page, largest size, and no title yet)
            if (page_num == 1 and not title and 
                size == max(heading_sizes) and 
                len(text.split()) <= 10):  # Titles are usually not too long
                title = text
                heading_sizes.remove(size)
                if (page.rect.height* 20/100) < heading['y_pos'] < (page.rect.height* 50/100):
                    break
                continue
            
            # Determine heading level based on size hierarchy
            try:
                level_index = heading_sizes.index(size)
                if level_index == 0:
                    level = "H1"
                elif level_index == 1:
                    level = "H2"
                elif level_index == 2:
                    level = "H3"
                else:
                    level = "H4"
                
                result_outline.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "font_size": size
                })
            except ValueError:
                # Fallback if size not in heading_sizes
                if size > avg_font_size * 1.3:
                    level = "H1"
                elif size > avg_font_size * 1.2:
                    level = "H2"
                elif size > avg_font_size * 1.1:
                    level = "H3"
                else:
                    level = "H4"
                
                result_outline.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "font_size": size
                })
    
    doc.close()
    
    # Post-process to remove obvious duplicates
    final_outline = []
    seen_texts = set()
    
    for item in result_outline:
        text_lower = item['text'].lower().strip()
        if text_lower not in seen_texts and len(text_lower) > 2:
            seen_texts.add(text_lower)
            final_outline.append(item)
    
    return {
        "title": title,
        "outline": final_outline,
        "stats": {
            "avg_font_size": avg_font_size,
            "body_font_size": body_font_size,
            "heading_sizes": heading_sizes
        }
    }

def process_pdf_file(input_dir, output_dir):
    """Process all PDF files in the input directory"""
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist!")
        return
        
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for filename in pdf_files:
        print(f"\nProcessing {filename}...")
        input_path = os.path.join(input_dir, filename)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            
            data = extract_headings(input_path)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"✓ Extracted {len(data['outline'])} headings from {filename}")
            print(f"✓ Title: {data['title']}")
            
            # Preview headings
            print("✓ Headings found:")
            for heading in data['outline'][:10]:  # Show first 10
                print(f"  {heading['level']}: {heading['text']} (Page {heading['page']})")
            if len(data['outline']) > 10:
                print(f"  ... and {len(data['outline']) - 10} more")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()
    input_dir = "C:/Users/abhiv/Documents/Adobe/my_pdfs"
    output_dir = "C:/Users/abhiv/Documents/Adobe/output_mine"
    
    process_pdf_file(input_dir, output_dir)
    print("\nx----- END -----x")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")