@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════╗
echo ║   🚀 AI 数据分析助手 - 一键启动             ║
echo ╚══════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装
    pause
    exit /b 1
)

REM 检查 Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Node.js，请先安装
    pause
    exit /b 1
)

echo ✅ 环境检查通过
echo.
echo 🚀 正在启动后端服务器 (http://localhost:8000)...
start "后端 API" cmd /k "python other\backend_api.py"

echo ⏳ 等待后端启动...
timeout /t 3 /nobreak >nul

echo.
echo 🚀 正在启动前端界面 (http://localhost:3000)...
cd frontend
start "前端界面" cmd /k "npm run dev"
cd ..

echo.
echo ═══════════════════════════════════════════════
echo   ✅ 启动完成！
echo ═══════════════════════════════════════════════
echo.
echo 📊 前端地址: http://localhost:3000
echo 🔧 后端 API: http://localhost:8000
echo.
echo 💡 使用说明:
echo    1. 浏览器会自动打开，或手动访问 http://localhost:3000
echo    2. 上传 CSV 文件
echo    3. 用自然语言提问开始智能分析
echo.
echo ⚠️  关闭此窗口不会停止服务器
echo    需要手动关闭后端和前端的窗口
echo.
pause

