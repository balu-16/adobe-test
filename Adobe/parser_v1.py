import fitz  # PyMuPDF
import os
import json
import re
import time
from statistics import mean, median, mode

ROUND_DIGIT = 2

def clean_repeated_text(text):
    """Removes repeated word fragments and cleans heading text."""
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.I)  # remove repeated words
    text = re.sub(r'([A-Za-z]{2,})\1+', r'\1', text)              # repeated fragments
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_likely_heading(text, size, avg_size, body_size):
    """Uses size, text features and context to decide if text is likely a heading."""
    if len(text) < 2 or len(text) > 150: return False
    if size < max(avg_size*1.02, body_size*1.08): return False   # stricter thresholding
    if re.match(r'^\d+$', text): return False
    if text.lower() in ('table', 'figure', 'continued'): return False
    if text.endswith('.'): return False
    if re.match(r'^page \d+$', text.lower()): return False
    if re.search(r'www\.|@|\.\w{2,3}', text): return False
    if len(text) < 8 and (not text.isupper() and not text.istitle()): return False
    # Heading-like: high-case, centered (TODO: Add if available), short, has numbering etc
    return True

def get_body_font_size(font_sizes):
    """Returns the most commonly used font size (by char count)."""
    return max(font_sizes, key=font_sizes.get)

def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    font_stats = {}
    total_text_len = 0
    total_size = 0

    # First pass: gather font statistics
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    t = span['text'].strip()
                    if not t: continue
                    size = round(span['size'], ROUND_DIGIT)
                    font_stats[size] = font_stats.get(size, 0) + len(t)
                    total_size += size * len(t)
                    total_text_len += len(t)

    if not font_stats: return {"title": "", "outline": []}
    avg_size = total_size/total_text_len if total_text_len else 10
    body_size = get_body_font_size(font_stats)
    heading_sizes = [sz for sz in sorted(font_stats.keys(), reverse=True) if sz >= body_size*1.25]

    title = ""
    outline = []
    duplicate_filter = set()

    # Second pass: extract headings per page (with multi-line/continuations)
    for page_num, page in enumerate(doc, 1):
        page_headings = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            
            x0, y0, x1, y1 = block['bbox']
            page_height = page.rect.height
            if y1 < 80 or y0 > page_height - 80:
                continue
            for line in block.get("lines", []):
                # Combine all spans (sometimes a heading is split into several spans)
                line_text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                line_size = max(round(span["size"], ROUND_DIGIT) for span in line["spans"])
                if not line_text: continue

                cleaned = clean_repeated_text(line_text)
                if is_likely_heading(cleaned, line_size, avg_size, body_size):
                    page_headings.append({
                        "text": cleaned,
                        "size": line_size,
                        "y0": line["bbox"][1],
                        "y1": line["bbox"][3],
                        "page": page_num
                    })

        # Try to merge adjacent heading fragments (same size, close vertical distance)
        merged = []
        buffer = None
        for h in sorted(page_headings, key=lambda x: x['y0']):
            if buffer and (abs(h['y0'] - buffer['y1']) < 2.5 * (buffer['y1'] - buffer['y0'])) and (h['size'] == buffer['size']):
                buffer['text'] += ' ' + h['text']
            else:
                if buffer: merged.append(buffer)
                buffer = h
        if buffer: merged.append(buffer)

        # Assign levels, deduplicate, add to outline
        for h in merged:
            cleaned = h['text'].strip()
            if cleaned.lower() in duplicate_filter: continue
            duplicate_filter.add(cleaned.lower())
            # Level assignment based on font size rank
            try:
                level_idx = heading_sizes.index(h['size'])
            except ValueError:
                level_idx = 0
            level = {0: "H1", 1: "H2", 2: "H3"}.get(level_idx, "H4")

            if not title and level == 'H1' and 2 <= len(cleaned.split()) <= 10:
                title = cleaned
                continue
            outline.append({
                "level": level,
                "text": cleaned,
                "page": h['page'],
                "font_size": h['size']
            })

    doc.close()
    return {
        "title": title,
        "outline": outline
    }

# ... (rest of your batch-processing code unchanged) ...

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
    input_dir = "C:/Users/abhiv/Documents/Adobe/pdfs"
    output_dir = "C:/Users/abhiv/Documents/Adobe/output_mine"
    
    process_pdf_file(input_dir, output_dir)
    print("\nx----- END -----x")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")