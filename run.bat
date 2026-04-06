@echo off
echo ============================================
echo  StockAI — AI Inventory System
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from python.org
    pause & exit /b 1
)
echo [OK] Python found

echo.
echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause & exit /b 1
)
echo [OK] Dependencies installed

echo.
echo [2/2] Starting server...
echo.
echo ============================================
echo  Open browser: http://localhost:5000
echo  Ctrl+C to stop
echo ============================================
echo.
python backend\app.py
pause
