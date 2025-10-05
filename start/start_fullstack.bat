@echo off
echo ========================================
echo   AI 数据分析助手 - 全栈启动脚本
echo ========================================
echo.

cd /d "%~dp0"

echo 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python
    echo 请先安装 Python: https://python.org/
    pause
    exit /b 1
)

echo ✅ Python 环境正常

echo.
echo 检查 Node.js 环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Node.js
    echo 请先安装 Node.js: https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js 环境正常

echo.
echo 检查 Python 依赖...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo 📦 安装 Python 依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Python 依赖安装失败
        pause
        exit /b 1
    )
)

echo ✅ Python 依赖正常

echo.
echo 检查前端依赖...
cd frontend
if not exist "node_modules" (
    echo 📦 安装前端依赖...
    npm install
    if errorlevel 1 (
        echo ❌ 前端依赖安装失败
        pause
        exit /b 1
    )
)
cd ..

echo ✅ 前端依赖正常

echo.
echo 🚀 启动后端 API 服务器...
if exist ".venv\Scripts\activate.bat" (
    start "Backend API" cmd /k ".venv\Scripts\activate.bat && python other/backend_api.py"
) else if exist "venv\Scripts\activate.bat" (
    start "Backend API" cmd /k "venv\Scripts\activate.bat && python other/backend_api.py"
) else (
    start "Backend API" cmd /k "python other/backend_api.py"
)

echo.
echo ⏳ 等待后端启动...
timeout /t 3 /nobreak >nul

echo.
echo 🚀 启动前端开发服务器...
cd frontend
start "Frontend Dev Server" cmd /k "npm run dev"

echo.
echo ========================================
echo   🎉 全栈应用启动完成！
echo ========================================
echo.
echo 📊 前端地址: http://localhost:3000
echo 🔧 后端 API: http://localhost:8000
echo.
echo 使用说明:
echo 1. 打开浏览器访问 http://localhost:3000
echo 2. 上传 CSV 文件开始数据分析
echo 3. 用自然语言提问进行智能分析
echo.
echo 按任意键关闭此窗口...
pause >nul
