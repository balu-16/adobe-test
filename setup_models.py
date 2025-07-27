#!/usr/bin/env python3
"""
Setup script to download and prepare models for the Document Intelligence system.
Run this script first to ensure all models are available locally.
"""

import os
import sys
from pathlib import Path

def setup_models():
    """Download and save models locally."""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print("Setting up models for offline usage...")
    
    try:
        # Import after ensuring dependencies are installed
        from sentence_transformers import SentenceTransformer
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        # Download and save sentence transformer
        print("1. Downloading sentence-transformers/all-MiniLM-L6-v2...")
        embedding_path = models_dir / "all-MiniLM-L6-v2"
        if not embedding_path.exists() or not any(embedding_path.iterdir()):
            try:
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                model.save(str(embedding_path))
                print("   ✓ Embedding model saved")
            except Exception as e:
                print(f"   ❌ Error downloading embedding model: {str(e)}")
                return False
        else:
            print("   ✓ Embedding model already exists")
        
        # Download and save T5 model
        print("2. Downloading t5-small...")
        t5_path = models_dir / "t5-small"
        if not t5_path.exists() or not any(t5_path.iterdir()):
            try:
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                tokenizer.save_pretrained(str(t5_path))
                model.save_pretrained(str(t5_path))
                print("   ✓ T5 model saved")
            except Exception as e:
                print(f"   ❌ Error downloading T5 model: {str(e)}")
                return False
        else:
            print("   ✓ T5 model already exists")
        
        print("\n🎉 All models are ready for offline usage!")
        print(f"Models saved in: {models_dir.absolute()}")
        
        # Calculate total size
        try:
            total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
            print(f"Total model size: {total_size / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"Could not calculate model size: {str(e)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_models()
    sys.exit(0 if success else 1)