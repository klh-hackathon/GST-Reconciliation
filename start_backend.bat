@echo off
echo ============================================
echo   GST Recon Engine - Backend Startup
echo ============================================
echo.

:: Step 1: Start the backend
echo [1/2] Starting FastAPI backend on port 8000...
start /B cmd /c "cd /d %~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 >nul

:: Step 2: Start Cloudflare Tunnel
echo [2/2] Starting Cloudflare Tunnel...
echo.
echo    Your tunnel URL will appear below.
echo    Copy it and paste into frontend\js\config.js
echo    Then push to GitHub.
echo.
echo ============================================
cloudflared tunnel --url http://localhost:8000
