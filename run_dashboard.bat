@echo off
echo Starting Employee Review Analytics Dashboard...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if merged_reviews.json exists
if not exist "merged_reviews.json" (
    echo Error: merged_reviews.json not found
    echo Please make sure the data file is in the same directory
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing requirements...
pip install -r requirements.txt

REM Run the dashboard
echo.
echo Starting dashboard...
echo Open your browser and go to: http://localhost:8501
echo.
streamlit run dashboard.py

pause
