@echo off
echo Adobe India Hackathon 2025 - Round 1B Setup
echo ==========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH!
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo 1. Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Warning: Could not upgrade pip, continuing...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies!
    echo Trying with --no-cache-dir flag...
    pip install --no-cache-dir -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install dependencies even with --no-cache-dir
        pause
        exit /b 1
    )
)

echo.
echo 2. Creating required directories...
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "models" mkdir models

echo.
echo 3. Setting up models for offline usage...
python setup_models.py
if %errorlevel% neq 0 (
    echo Error setting up models!
    echo This might be due to network issues or dependency conflicts.
    echo You can try running 'python setup_models.py' manually later.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo.
echo Next steps:
echo 1. Place your PDF files in the 'input' folder
echo 2. Run: python main.py
echo 3. Follow the prompts to enter persona and job
echo 4. Check results in 'output/result.json'
echo.
echo Press any key to exit...
pause >nul