#!/bin/bash

echo "Adobe India Hackathon 2025 - Round 1B Setup"
echo "=========================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "Python version:"
python3 --version
echo

echo "1. Upgrading pip and installing dependencies..."
python3 -m pip install --upgrade pip

if ! pip install -r requirements.txt; then
    echo "Error installing dependencies!"
    echo "Trying with --no-cache-dir flag..."
    if ! pip install --no-cache-dir -r requirements.txt; then
        echo "Failed to install dependencies even with --no-cache-dir"
        exit 1
    fi
fi

echo
echo "2. Creating required directories..."
mkdir -p input output models

echo
echo "3. Setting up models for offline usage..."
if ! python3 setup_models.py; then
    echo "Error setting up models!"
    echo "This might be due to network issues or dependency conflicts."
    echo "You can try running 'python3 setup_models.py' manually later."
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo
echo "Next steps:"
echo "1. Place your PDF files in the 'input' folder"
echo "2. Run: python3 main.py"
echo "3. Follow the prompts to enter persona and job"
echo "4. Check results in 'output/result.json'"
echo