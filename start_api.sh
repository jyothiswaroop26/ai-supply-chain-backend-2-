#!/bin/bash
# FastAPI Server Startup Script for Linux/Mac
# This script starts the AI Supply Chain Backend API

echo ""
echo "========================================"
echo "FastAPI Server Startup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    exit 1
fi

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Check if virtual environment exists (optional)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Using .env.example as reference."
    echo "Please create .env file with your configuration."
fi

# Install requirements if needed
echo ""
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Start the server
echo ""
echo "Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Swagger docs at: http://localhost:8000/docs"
echo ""

python -m api.main
