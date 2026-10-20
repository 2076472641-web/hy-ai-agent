@echo off
cd /d "%~dp0"

echo ============================================
echo   AI Desensitizer Tool
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

python -c "import flask, requests" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo Starting server, browser will open automatically...
python app.py
pause
