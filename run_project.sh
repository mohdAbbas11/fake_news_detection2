#!/bin/bash

echo "===== Fake News Detection Project ====="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install venv package."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run the project
echo "Running the project..."
python run_project.py "$@"

# Deactivate virtual environment
deactivate

echo ""
echo "===== Project execution completed ====="