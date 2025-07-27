# Adobe India Hackathon 2025 - Round 1B
## Persona-Driven Document Intelligence

A sophisticated document intelligence system that extracts and analyzes information from PDF documents based on user-defined personas and specific job requirements.

## Features

- **Persona-Driven Analysis**: Tailors document processing based on user-specified personas
- **Interactive Input**: Prompts users for persona and job requirements
- **Multi-Document Processing**: Handles multiple PDF files simultaneously
- **Intelligent Section Extraction**: Identifies and extracts relevant document sections
- **Semantic Analysis**: Uses advanced NLP models for content understanding
- **Structured Output**: Generates comprehensive JSON reports with metadata and analysis

## Quick Start

### Windows
```bash
setup.bat
```

### Linux/macOS
```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Quick Demo
For a guided demonstration with sample scenarios:
```bash
python demo.py
```

### Manual Usage
1. **Place PDF files** in the `input/` directory
2. **Run the system**:
   ```bash
   python main.py
   ```
3. **Follow the prompts**:
   - Enter your persona (e.g., "PhD Researcher in Computational Biology")
   - Enter the job to be done (e.g., "Prepare a literature review on machine learning methods")
4. **Check results** in `output/result.json`

## Manual Setup

If the automatic setup doesn't work, follow these steps:

1. **Install Python 3.8+**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create directories**:
   ```bash
   mkdir input output models
   ```
4. **Download models**:
   ```bash
   python setup_models.py
   ```

## Project Structure

```
├── main.py              # Main application entry point
├── demo.py              # Interactive demo script
├── setup_models.py      # Model download and setup
├── requirements.txt     # Python dependencies
├── setup.bat           # Windows setup script
├── setup.sh            # Linux/macOS setup script
├── README.md           # This file
├── input/              # Place PDF files here
│   └── sample_research_paper.pdf  # Sample document for demo
├── output/             # Generated results
└── models/             # Downloaded AI models
    ├── all-MiniLM-L6-v2/
    └── t5-small/
```

## 🧠 How It Works

### 1. PDF Processing
- Uses **PyMuPDF (fitz)** for robust PDF text extraction
- Intelligently detects section headers using regex patterns
- Extracts structured content with page numbers

### 2. Semantic Ranking
- Uses **sentence-transformers/all-MiniLM-L6-v2** for embeddings
- Computes cosine similarity between job description and sections
- Ranks top 3-10 most relevant sections

### 3. Summarization
- Uses **HuggingFace t5-small** model for summarization
- Adds required "summarize: " prefix for T5
- Generates concise summaries for each selected section

### 4. Output Generation
- Creates JSON in the exact required format
- Includes metadata, ranked sections, and summaries
- Saves timestamp using ISO format

## ⚡ Performance Features

- **Parallel Processing**: Efficient section extraction
- **Smart Caching**: Models loaded once and reused
- **Memory Optimization**: Processes sections in batches
- **Error Handling**: Robust error recovery and fallbacks

## 🔧 Configuration

You can modify the persona and job description in `main.py`:

```python
PERSONA = "PhD Researcher in Computational Biology"
JOB_TO_BE_DONE = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
```

## 📊 Expected Output Format

```json
{
  "metadata": {
    "documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review...",
    "timestamp": "2025-07-26T17:00:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page": 2,
      "section_title": "Graph Neural Networks in Drug Target Interaction",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "page": 2,
      "refined_text": "This section reviews GNN models used in predicting drug-target interactions..."
    }
  ]
}
```

## 🛠️ Technical Details

### Models Used
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
- **Summarization Model**: t5-small (~240MB)
- **Total Size**: < 400MB

### Performance Benchmarks
- **5 PDFs**: < 60 seconds
- **10 PDFs**: < 90 seconds
- **Memory Usage**: < 2GB RAM

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 2GB free disk space
- Windows/Linux/macOS compatible

## 🐛 Troubleshooting

### Common Issues

1. **"No PDF files found"**
   - Ensure PDFs are in `./input/` folder
   - Check file extensions are `.pdf`

2. **Model download errors**
   - Check internet connection for first run
   - Models are cached locally after first download

3. **Memory errors**
   - Reduce number of PDFs processed at once
   - Close other applications to free RAM

## 🔧 Troubleshooting

### Common Issues

**ImportError: cannot import name 'cached_download' from 'huggingface_hub'**
- This is a version compatibility issue
- Solution: Use the provided `requirements.txt` with tested versions
- Or run: `pip install huggingface-hub==0.19.4 transformers==4.36.2`

**Models not downloading**
- Check internet connection
- Run `python setup_models.py` manually
- Ensure you have ~320MB free disk space

**Empty output or no sections found**
- Ensure PDF files are in `./input/` directory
- Check that PDFs contain readable text (not just images)
- Verify PDFs are not password-protected

**Performance issues**
- Ensure you have at least 4GB RAM available
- Close other applications to free up memory
- Consider processing fewer PDFs at once

### Debug Mode
To enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License
This solution is created for the Adobe India Hackathon 2025 - Round 1B.

---
**Built with ❤️ for Adobe India Hackathon 2025**