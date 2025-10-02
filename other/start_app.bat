@echo off
echo ========================================
echo   Data Analysis AI Agent
echo   Starting Backend and Frontend
echo ========================================
echo.

REM Start Flask Backend
echo [1/2] Starting Flask Backend on port 8000...
start "Flask Backend" cmd /k "python backend_api.py"
timeout /t 3 /nobreak > nul

REM Start Next.js Frontend
echo [2/2] Starting Next.js Frontend on port 3000...
cd frontend
start "Next.js Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo   Services Started!
echo ========================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WindowTitle eq Flask Backend*" /T /F
taskkill /FI "WindowTitle eq Next.js Frontend*" /T /F

