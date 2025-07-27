import fitz  # PyMuPDF
import os
import json
import re
import time


ROUND_DIGIT = 2

def is_likely_heading(text, size, avg_size):
    """Check if text is likely a heading based on various criteria"""
    text = text.strip()
    
    # Basic filters
    if len(text) < 3 or len(text) > 200:
        return False
    
    # Size should be notably larger than average
    if size < avg_size:
        return False
    
    # Filter out common non-heading patterns
    if re.match(r'^\d+$', text):  # Just numbers
        return False
    if re.match(r'^[^\w\s]+$', text):  # Just punctuation
        return False
    if text.lower() in ['page', 'table', 'figure', 'chart']:
        return False
    
    # Good heading indicators
    if text[0].isupper() and not text.islower():  # Starts with capital
        return True
    if text.isupper() and len(text) > 10:  # All caps (likely heading)
        return True
    
    return True

def remove_repeated_chunks_in_line(text):
    def collapse_repeats(line):
        # Match repeating substrings of length ≥ 2 that occur consecutively
        # Example: 'quest f quest f' -> 'quest f'
        return re.sub(r'(\b\S{4,}.*?)(\s+\1)+', r'\1', line)

    lines = text.splitlines()
    cleaned = [collapse_repeats(line) for line in lines]
    return "\n".join(cleaned)


def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    result_outline = []
    title = ""
    
    # First pass: collect all font sizes to determine hierarchy
    all_font_sizes = []
    total_font_size = 0
    len_font = 0
    


    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        

        


        for block in blocks:
            # Skip headers and footers
            x0, y0, x1, y1 = block['bbox']
            if y1 < 100 or y0 > page.rect.height - 100:
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], ROUND_DIGIT)
                    total_font_size += size * len(text)
                    len_font += len(text)

                    if size not in all_font_sizes:
                        all_font_sizes.append(size)
    
    # Calculate average font size (likely body text)
    avg_font_size = total_font_size / len_font
    print(f"Average Font size is : {avg_font_size}")

    # Get unique font sizes for heading hierarchy
    unique_sizes = sorted(list(set(all_font_sizes)), reverse=True)
    heading_sizes = [size for size in unique_sizes if size > avg_font_size * 1.1]
    
    # Second pass: extract headings
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        heading_candidates= []
        page_headings = {}  # size -> [texts]
        
        for block in blocks:
            # Skip headers and footers
            x0, y0, x1, y1 = block['bbox']
            if y1 < 100 or y0 > page.rect.height - 100:
                continue
                
            for line in block.get("lines", []):
                line_text = ""
                line_size = 0
                
                # Combine spans in the same line
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], ROUND_DIGIT)
                    
                    if text and size >= avg_font_size:
                        line_text += text + " "
                        line_bbox = line['bbox']
                        line_size = max(line_size, size)
                
                line_text = line_text.strip()
                
                # Check if this line is a potential heading
                if line_text and line_size in heading_sizes:
                    if is_likely_heading(line_text, line_size, avg_font_size):
                        heading_candidates.append({
                            'text': line_text,
                            'size': line_size,
                            'bbox': line_bbox,
                            'y_pos': line_bbox[1]  # top y coordinate
                        })
        
        # Group nearby heading candidates of the same size
        grouped_headings = []
        heading_candidates.sort(key=lambda x: x['y_pos'])
        
        i = 0
        while i < len(heading_candidates):
            current = heading_candidates[i]
            combined_text = current['text']
            current_size = current['size']
            
            # Look for continuation lines (same size, close proximity)
            j = i + 1
            while j < len(heading_candidates):
                next_candidate = heading_candidates[j]
                
                # Check if next line is continuation:
                # 1. Same font size
                # 2. Within reasonable vertical distance (e.g., 2-3 line heights)
                # 3. Similar or overlapping horizontal position
                y_distance = abs(next_candidate['y_pos'] - current['bbox'][3])  # distance from bottom of current to top of next
                line_height = current['bbox'][3] - current['bbox'][1]
                
                if (next_candidate['size'] == current_size and 
                    y_distance < line_height * 2.5):  # Similar x position (within 50 units)
                    
                    combined_text += " " + next_candidate['text']
                    current = next_candidate  # Update current for next iteration
                    heading_candidates.pop(j)  # Remove the combined line
                else:
                    break
            
            # Clean up the combined text
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            
            # Final check if the combined text is still a good heading
            if is_likely_heading(combined_text, current_size, avg_font_size):
                grouped_headings.append({
                    'text': remove_repeated_chunks_in_line(combined_text),
                    'size': current_size,
                    'y_pos': heading_candidates[i]['y_pos'] if i < len(heading_candidates) else current['y_pos']
                })
            
            i += 1
        
        # Sort by vertical position and process
        grouped_headings.sort(key=lambda x: x['y_pos'])
        
        # Process headings for this page
        for heading in grouped_headings:
            text = heading['text']
            size = heading['size']
            level_index = heading_sizes.index(size)
            
            if page_num == 1 and level_index == 0 and not title:
                # First largest heading on first page is likely the title
                title = text
                for heading in grouped_headings:
                    heading_sizes.remove(heading['size'])
                break
            else:
                # Determine heading level
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
                    "page": page_num
                })
    
    doc.close()
    
    return {
        "title": title,
        "outline": result_outline,
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
        print(f"Processing {filename}...")
        input_path = os.path.join(input_dir, filename)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            
            data = extract_headings(input_path)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"✓ Extracted {len(data['outline'])} headings from {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    input_dir = "C:/Users/abhiv/Documents/Adobe/pdfs"
    output_dir = "C:/Users/abhiv/Documents/Adobe/output_mine"
    
    process_pdf_file(input_dir, output_dir)
    print("x----- END -----x")
    print(f"Time taken: {time.time() - start_time}")