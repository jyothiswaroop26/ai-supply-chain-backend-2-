@echo off
REM FastAPI Server Startup Script for Windows
REM This script starts the AI Supply Chain Backend API

echo.
echo ========================================
echo FastAPI Server Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Navigate to backend directory
cd /d "%~dp0backend" || exit /b 1

REM Check if virtual environment exists (optional)
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found. Using .env.example as reference.
    echo Please create .env file with your configuration.
)

REM Install requirements if needed
echo.
echo Checking dependencies...
pip install -q -r requirements.txt

REM Start the server
echo.
echo Starting API server...
echo API will be available at: http://localhost:8000
echo Swagger docs at: http://localhost:8000/docs
echo.

python -m api.main

pause
