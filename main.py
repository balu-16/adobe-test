#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Round 1B: Persona-Driven Document Intelligence
High-performance Python solution for intelligent PDF processing and summarization
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIntelligence:
    def __init__(self, models_dir: str = "./models"):
        """Initialize the Document Intelligence system with local models."""
        self.models_dir = Path(models_dir)
        self.embedding_model = None
        self.summarizer = None
        self.tokenizer = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required models from local storage."""
        print("Loading models...")
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)
        
        try:
            # Load sentence transformer for embeddings
            embedding_path = self.models_dir / "all-MiniLM-L6-v2"
            if embedding_path.exists() and any(embedding_path.iterdir()):
                print("Loading existing embedding model...")
                self.embedding_model = SentenceTransformer(str(embedding_path))
            else:
                print("Downloading embedding model...")
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model.save(str(embedding_path))
            
            # Load T5 model for summarization
            t5_path = self.models_dir / "t5-small"
            if t5_path.exists() and any(t5_path.iterdir()):
                print("Loading existing T5 model...")
                self.tokenizer = T5Tokenizer.from_pretrained(str(t5_path))
                model = T5ForConditionalGeneration.from_pretrained(str(t5_path))
                self.summarizer = pipeline("summarization", 
                                         model=model, 
                                         tokenizer=self.tokenizer,
                                         device=-1)  # CPU only
            else:
                print("Downloading T5 model...")
                self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                self.tokenizer.save_pretrained(str(t5_path))
                model.save_pretrained(str(t5_path))
                self.summarizer = pipeline("summarization", 
                                         model=model, 
                                         tokenizer=self.tokenizer,
                                         device=-1)
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load required models: {str(e)}")
    
    def extract_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract meaningful content chunks from PDF using improved parsing."""
        content_chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Split text into meaningful paragraphs and sections
                paragraphs = self._extract_meaningful_paragraphs(text)
                
                for para in paragraphs:
                    if len(para['content'].split()) >= 10:  # Only meaningful content
                        content_chunks.append({
                            'document': os.path.basename(pdf_path),
                            'page_number': page_num + 1,
                            'section_title': para['title'],
                            'content': para['content'],
                            'word_count': len(para['content'].split())
                        })
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
        
        return content_chunks
    
    def _extract_meaningful_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """Extract meaningful paragraphs with proper titles."""
        paragraphs = []
        lines = text.split('\n')
        
        current_content = ""
        current_title = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line could be a title/header
            if self._is_potential_title(line):
                # Save previous paragraph if it has substantial content
                if current_content.strip() and len(current_content.split()) >= 10:
                    paragraphs.append({
                        'title': current_title or self._generate_title_from_content(current_content),
                        'content': current_content.strip()
                    })
                
                current_title = line
                current_content = ""
            else:
                current_content += line + " "
        
        # Add the last paragraph
        if current_content.strip() and len(current_content.split()) >= 10:
            paragraphs.append({
                'title': current_title or self._generate_title_from_content(current_content),
                'content': current_content.strip()
            })
        
        return paragraphs
    
    def _is_potential_title(self, line: str) -> bool:
        """Improved title detection."""
        line = line.strip()
        
        if len(line) < 3 or len(line) > 100:
            return False
        
        # Common title patterns
        title_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # "INTRODUCTION"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Introduction"
            r'^\d+\.\d+',  # "1.1 Background"
            r'^(Abstract|Introduction|Conclusion|References|Methodology|Results|Discussion|Overview|Summary)$',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
            r'^[A-Z][a-z]+(\s+(and|or|of|in|for|with|to)\s+[A-Z][a-z]+)*$',  # "Tips and Tricks"
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True
        
        # Heuristic: short lines that start with capital and don't end with punctuation
        words = line.split()
        if (len(words) <= 8 and 
            line[0].isupper() and 
            not line.endswith(('.', ',', ';', ':', '!', '?')) and
            not any(word.islower() and len(word) > 3 for word in words[:2])):  # Avoid sentences
            return True
        
        return False
    
    def _generate_title_from_content(self, content: str) -> str:
        """Generate a title from content if no explicit title found."""
        words = content.split()[:8]
        title = ' '.join(words)
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    
    def rank_content_by_persona_and_job(self, content_chunks: List[Dict], persona: str, job_to_be_done: str, top_k: int = 5) -> List[Dict]:
        """Rank content by relevance to persona, job, and PDF file names using semantic similarity."""
        if not content_chunks:
            return []
        
        # Create a combined query that includes both persona and job context
        persona_job_query = f"As a {persona}, I need to {job_to_be_done}"
        
        # Create embeddings for the persona-job query
        query_embedding = self.embedding_model.encode([persona_job_query])
        
        # Create embeddings for all content chunks with enhanced context
        content_texts = []
        for chunk in content_chunks:
            # Extract meaningful keywords from PDF filename
            pdf_name = chunk['document']
            filename_context = self._extract_filename_context(pdf_name)
            
            # Combine filename context, title, and content for better matching
            combined_text = f"{filename_context} {chunk['section_title']} {chunk['content']}"
            content_texts.append(combined_text)
        
        content_embeddings = self.embedding_model.encode(content_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, content_embeddings)[0]
        
        # Add similarity scores and filename relevance
        for i, chunk in enumerate(content_chunks):
            base_similarity = float(similarities[i])
            
            # Calculate filename relevance boost
            filename_boost = self._calculate_filename_relevance(
                chunk['document'], persona, job_to_be_done
            )
            
            # Combine similarity with filename relevance (weighted)
            chunk['similarity_score'] = base_similarity + (filename_boost * 0.3)  # 30% weight for filename
            chunk['filename_relevance'] = filename_boost
        
        # Sort by combined score (descending) and take top_k
        ranked_content = sorted(content_chunks, key=lambda x: x['similarity_score'], reverse=True)
        
        # Add importance rank
        for i, chunk in enumerate(ranked_content[:top_k]):
            chunk['importance_rank'] = i + 1
        
        return ranked_content[:top_k]
    
    def _extract_filename_context(self, pdf_filename: str) -> str:
        """Extract meaningful context from PDF filename."""
        # Remove file extension and clean filename
        name = pdf_filename.replace('.pdf', '').replace('.PDF', '')
        
        # Replace common separators with spaces
        name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Split into words and filter out common non-meaningful words
        words = name.split()
        meaningful_words = []
        
        skip_words = {'ideas', 'tips', 'guide', 'manual', 'document', 'file', 'pdf', 'part', 'section'}
        
        for word in words:
            word_clean = word.lower().strip()
            if len(word_clean) > 2 and word_clean not in skip_words:
                meaningful_words.append(word_clean)
        
        return ' '.join(meaningful_words)
    
    def _calculate_filename_relevance(self, pdf_filename: str, persona: str, job_to_be_done: str) -> float:
        """Calculate how relevant the PDF filename is to the persona and job."""
        filename_lower = pdf_filename.lower()
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        relevance_score = 0.0
        
        # Extract keywords from filename
        filename_keywords = self._extract_filename_context(pdf_filename).split()
        
        # Check for direct matches with persona keywords
        persona_keywords = persona_lower.split()
        for p_word in persona_keywords:
            if len(p_word) > 3:  # Skip short words
                for f_word in filename_keywords:
                    if p_word in f_word or f_word in p_word:
                        relevance_score += 0.4
        
        # Check for direct matches with job keywords
        job_keywords = job_lower.split()
        for j_word in job_keywords:
            if len(j_word) > 3:  # Skip short words like 'a', 'the', 'of'
                for f_word in filename_keywords:
                    if j_word in f_word or f_word in j_word:
                        relevance_score += 0.5
        
        # Special keyword matching for common domains
        domain_keywords = {
            'food': ['breakfast', 'lunch', 'dinner', 'meal', 'recipe', 'cooking', 'vegetarian', 'vegan'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'hotel', 'flight', 'tourism'],
            'business': ['business', 'corporate', 'company', 'management', 'strategy', 'finance'],
            'health': ['health', 'medical', 'wellness', 'fitness', 'nutrition', 'diet'],
            'education': ['education', 'learning', 'study', 'academic', 'research', 'university']
        }
        
        # Check if filename contains domain-specific keywords that match persona/job
        for domain, keywords in domain_keywords.items():
            if domain in persona_lower or domain in job_lower:
                for keyword in keywords:
                    if keyword in filename_lower:
                        relevance_score += 0.3
        
        # Normalize score to 0-1 range
        return min(relevance_score, 1.0)
    
    def generate_persona_specific_summaries(self, content_chunks: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """Generate persona-specific summaries that focus on relevant aspects for the given persona and job."""
        summaries = []
        
        for chunk in content_chunks:
            try:
                # Create persona-specific prompt for summarization
                content = chunk['content']
                pdf_filename = chunk.get('document', '')
                
                # Persona-specific processing with filename context
                refined_text = self._create_persona_specific_summary(content, persona, job_to_be_done, pdf_filename)
                
                summaries.append({
                    'document': chunk['document'],
                    'page_number': chunk['page_number'],
                    'refined_text': refined_text
                })
                
            except Exception as e:
                print(f"Error creating summary for {chunk.get('document', 'unknown')}: {str(e)}")
                # Fallback to original content (truncated)
                content = chunk.get('content', '')
                summaries.append({
                    'document': chunk.get('document', 'unknown'),
                    'page_number': chunk.get('page_number', 1),
                    'refined_text': content[:300] + "..." if len(content) > 300 else content
                })
        
        return summaries
    
    def _create_persona_specific_summary(self, content: str, persona: str, job_to_be_done: str, pdf_filename: str = "") -> str:
        """Create a summary tailored to the specific persona, job requirements, and PDF context using T5 model."""
        
        # Persona-specific keywords and focus areas
        persona_focus = {
            'travel planner': ['itinerary', 'activities', 'attractions', 'accommodation', 'transportation', 'budget', 'schedule'],
            'hr professional': ['forms', 'compliance', 'onboarding', 'documentation', 'procedures', 'requirements', 'process'],
            'food contractor': ['menu', 'ingredients', 'recipes', 'dietary', 'preparation', 'serving', 'nutrition', 'vegetarian', 'vegan', 'gluten-free'],
            'phd researcher': ['methodology', 'research', 'analysis', 'data', 'findings', 'literature', 'study'],
            'medical doctor': ['treatment', 'diagnosis', 'patient', 'clinical', 'medical', 'health', 'symptoms'],
            'data scientist': ['data', 'analysis', 'model', 'algorithm', 'statistics', 'machine learning', 'insights']
        }
        
        # Get focus keywords for this persona
        focus_keywords = persona_focus.get(persona.lower(), [])
        
        # Extract filename context for additional relevance
        filename_context = self._extract_filename_context(pdf_filename) if pdf_filename else ""
        filename_keywords = filename_context.split()
        
        # Clean and prepare content for summarization
        content = content.strip()
        if len(content) < 50:
            return content  # Too short to summarize meaningfully
        
        # Try T5 summarization with persona and filename context first
        try:
            # Limit content length for T5 model
            max_content_length = 400
            if len(content) > max_content_length:
                # Extract most relevant parts first
                sentences = content.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue
                    
                    # Score sentence based on persona relevance, job keywords, and filename context
                    score = 0
                    sentence_lower = sentence.lower()
                    job_lower = job_to_be_done.lower()
                    
                    # Check for persona-specific keywords
                    for keyword in focus_keywords:
                        if keyword in sentence_lower:
                            score += 2
                    
                    # Check for job-specific keywords
                    job_words = job_lower.split()
                    for word in job_words:
                        if len(word) > 3 and word in sentence_lower:
                            score += 3
                    
                    # Check for filename-related keywords (new)
                    for f_keyword in filename_keywords:
                        if len(f_keyword) > 3 and f_keyword in sentence_lower:
                            score += 2
                    
                    relevant_sentences.append((sentence, score))
                
                # Sort by relevance and take top sentences that fit within limit
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                selected_content = ""
                for sentence, score in relevant_sentences:
                    if len(selected_content + sentence) < max_content_length:
                        selected_content += sentence + ". "
                    else:
                        break
                
                content = selected_content.strip() if selected_content else content[:max_content_length]
            
            # Create summarization prompt with persona and filename context
            context_info = f"{persona}"
            if filename_context:
                context_info += f" working with {filename_context}"
            
            input_text = f"summarize for {context_info}: {content}"
            
            # Generate summary using T5
            summary_result = self.summarizer(
                input_text, 
                max_length=min(150, len(content.split()) + 20),  # Dynamic max length
                min_length=20, 
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-process summary to ensure it's persona-relevant and filename-aware
            if summary and len(summary) > 10:
                # Ensure summary ends properly
                if not summary.endswith('.'):
                    summary += '.'
                return summary
            else:
                raise Exception("Summary too short")
                
        except Exception as e:
            # Fallback: Create a manual summary from most relevant sentences
            sentences = content.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                # Score sentence based on persona relevance, job keywords, and filename context
                score = 0
                sentence_lower = sentence.lower()
                job_lower = job_to_be_done.lower()
                
                # Check for persona-specific keywords
                for keyword in focus_keywords:
                    if keyword in sentence_lower:
                        score += 2
                
                # Check for job-specific keywords
                job_words = job_lower.split()
                for word in job_words:
                    if len(word) > 3 and word in sentence_lower:
                        score += 3
                
                # Check for filename-related keywords (new)
                for f_keyword in filename_keywords:
                    if len(f_keyword) > 3 and f_keyword in sentence_lower:
                        score += 2
                
                if score > 0:
                    relevant_sentences.append((sentence, score))
            
            # Sort by relevance score and create summary
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                # Take top 2-3 most relevant sentences and create a coherent summary
                top_sentences = [sent[0] for sent in relevant_sentences[:2]]
                summary = '. '.join(top_sentences)
                if not summary.endswith('.'):
                    summary += '.'
                return summary
            else:
                # Final fallback
                return content[:150] + "..." if len(content) > 150 else content
    
    def process_documents(self, input_dir: str, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """Main processing pipeline with persona-specific analysis."""
        start_time = time.time()
        
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {input_dir}")
            # Return empty result structure
            return {
                "metadata": {
                    "input_documents": [],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Extract content from all PDFs
        all_content = []
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            content_chunks = self.extract_pdf_content(str(pdf_file))
            all_content.extend(content_chunks)
        
        print(f"Extracted {len(all_content)} content chunks")
        
        # Rank content by persona and job relevance
        top_content = self.rank_content_by_persona_and_job(all_content, persona, job_to_be_done, top_k=5)
        
        print(f"Found {len(top_content)} relevant sections")
        
        # Generate persona-specific summaries
        summaries = self.generate_persona_specific_summaries(top_content, persona, job_to_be_done)
        
        # Prepare output in the expected format
        result = {
            "metadata": {
                "input_documents": [pdf.name for pdf in pdf_files],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": chunk['document'],
                    "section_title": chunk['section_title'],
                    "importance_rank": chunk['importance_rank'],
                    "page_number": chunk['page_number']
                }
                for chunk in top_content
            ],
            "subsection_analysis": summaries
        }
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return result


def main():
    """Main execution function."""
    print("Adobe India Hackathon 2025 - Round 1B")
    print("Persona-Driven Document Intelligence")
    print("=" * 50)
    
    # Configuration
    INPUT_DIR = "./input"
    OUTPUT_DIR = "./output"
    MODELS_DIR = "./models"
    
    # Create directories if they don't exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check if PDF files exist
    input_path = Path(INPUT_DIR)
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {INPUT_DIR}")
        print("Please place your PDF files in the input folder and run again.")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    # Get user input for persona and job
    print("Please provide the following information:")
    print("-" * 40)
    
    persona = input("Enter the persona (e.g., 'PhD Researcher in Computational Biology'): ").strip()
    if not persona:
        persona = "Researcher"
        print(f"Using default persona: {persona}")
    
    print()
    job_to_be_done = input("Enter the job to be done (e.g., 'Prepare a literature review on machine learning methods'): ").strip()
    if not job_to_be_done:
        job_to_be_done = "Extract and summarize relevant information"
        print(f"Using default job: {job_to_be_done}")
    
    print()
    print("Starting document processing...")
    print("=" * 50)
    
    try:
        # Initialize the system
        print("Initializing Document Intelligence...")
        doc_intel = DocumentIntelligence(models_dir=MODELS_DIR)
        
        # Process documents
        result = doc_intel.process_documents(INPUT_DIR, persona, job_to_be_done)
        
        # Create a unique output filename based on persona and job
        persona_clean = "".join(c for c in persona if c.isalnum() or c in (' ', '-', '_')).rstrip()
        persona_clean = persona_clean.replace(' ', '_').lower()
        
        # Get first few words of job for filename
        job_words = job_to_be_done.split()[:3]  # First 3 words
        job_clean = "_".join(word.lower() for word in job_words if word.isalnum())
        
        # Create timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_filename = f"{persona_clean}_{job_clean}_{timestamp}.json"
        output_file = Path(OUTPUT_DIR) / output_filename
        
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print()
        print("Processing completed successfully!")
        print("=" * 50)
        print(f"Results saved to: {output_file}")
        print(f"Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"Relevant sections found: {len(result['extracted_sections'])}")
        print(f"Summaries generated: {len(result['subsection_analysis'])}")
        print()
        print("Document intelligence analysis complete!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())