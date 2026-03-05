@echo off
REM Quick Setup and Launch Script for Legacy Code Challenge System

echo ================================================
echo Legacy Code Challenge - Quick Start
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo ================================================
echo Setup complete!
echo ================================================
echo.
echo You can now:
echo   1. Generate a challenge: python main.py https://github.com/user/repo.git
echo   2. Launch student interface: python student_interface.py
echo.
echo Starting student interface in 3 seconds...
timeout /t 3 /nobreak >nul

python student_interface.py

pause
