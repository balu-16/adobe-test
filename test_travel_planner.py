#!/usr/bin/env python3
"""
Test script for Travel Planner persona
"""

import sys
import os
from pathlib import Path
import json

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DocumentIntelligence

def test_travel_planner():
    """Test the system with Travel Planner persona"""
    
    # Configuration
    INPUT_DIR = "./input"
    OUTPUT_DIR = "./output"
    MODELS_DIR = "./models"
    
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test parameters
    persona = "Travel Planner"
    job_to_be_done = "Plan a trip of 4 days for a group of 10 college friends."
    
    print(f"Testing with persona: {persona}")
    print(f"Job to be done: {job_to_be_done}")
    print("=" * 50)
    
    try:
        # Initialize the system
        doc_intel = DocumentIntelligence(models_dir=MODELS_DIR)
        
        # Process documents
        result = doc_intel.process_documents(INPUT_DIR, persona, job_to_be_done)
        
        # Save result
        output_file = Path(OUTPUT_DIR) / "travel_planner_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"Relevant sections found: {len(result['extracted_sections'])}")
        print(f"Summaries generated: {len(result['subsection_analysis'])}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

if __name__ == "__main__":
    test_travel_planner()