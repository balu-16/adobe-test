#!/usr/bin/env python3
"""
Demo Script for Adobe India Hackathon 2025 - Round 1B
Persona-Driven Document Intelligence System

This script demonstrates the system's capabilities with sample inputs.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_demo():
    """Run a demonstration of the document intelligence system."""
    
    print("Adobe India Hackathon 2025 - Round 1B")
    print("Persona-Driven Document Intelligence - DEMO")
    print("=" * 60)
    print()
    
    # Check if sample PDF exists
    sample_pdf = Path("./input/sample_research_paper.pdf")
    if not sample_pdf.exists():
        print("ERROR: Sample PDF not found!")
        print("Please ensure sample_research_paper.pdf is in the input folder")
        return 1
    
    print("Sample document found: sample_research_paper.pdf")
    print("File size: {:.1f} KB".format(sample_pdf.stat().st_size / 1024))
    print()
    
    # Demo scenarios
    demo_scenarios = [
        {
            "persona": "PhD Researcher in Machine Learning",
            "job": "Extract methodology and performance metrics for literature review"
        },
        {
            "persona": "Medical Doctor",
            "job": "Identify clinical applications and diagnostic accuracy results"
        },
        {
            "persona": "Data Scientist",
            "job": "Find dataset information and evaluation criteria"
        }
    ]
    
    print("Available Demo Scenarios:")
    print("-" * 40)
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"{i}. Persona: {scenario['persona']}")
        print(f"   Job: {scenario['job']}")
        print()
    
    # Get user choice
    try:
        choice = input("Select a demo scenario (1-3) or press Enter for interactive mode: ").strip()
        
        if choice in ['1', '2', '3']:
            scenario = demo_scenarios[int(choice) - 1]
            persona = scenario['persona']
            job = scenario['job']
            print(f"\nSelected Persona: {persona}")
            print(f"Selected Job: {job}")
        else:
            print("\nInteractive Mode - Enter your own persona and job:")
            persona = input("Enter persona: ").strip()
            job = input("Enter job to be done: ").strip()
            
            if not persona or not job:
                print("ERROR: Both persona and job are required!")
                return 1
        
        print("\nStarting document processing...")
        print("=" * 60)
        
        # Prepare input for main.py
        user_input = f"{persona}\n{job}\n"
        
        # Run main.py with the inputs
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        stdout, stderr = process.communicate(input=user_input)
        
        if process.returncode == 0:
            print(stdout)
            print("\nDemo completed successfully!")
            print("Check the output/ directory for your results file")
            print("(Filename includes persona, job, and timestamp for uniqueness)")
        else:
            print(f"ERROR occurred: {stderr}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user")
        return 0
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = run_demo()
    sys.exit(exit_code)