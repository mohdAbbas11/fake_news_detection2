@echo off
echo ===== Fake News Detection Project =====
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment. Please install venv package.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Run the project
echo Running the project...
python run_project.py %*

REM Deactivate virtual environment
call .venv\Scripts\deactivate.bat

echo.
echo ===== Project execution completed =====
pause