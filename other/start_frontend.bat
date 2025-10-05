@echo off
echo ========================================
echo   AI 数据分析助手 - 前端启动脚本
echo ========================================
echo.

cd /d "%~dp0"

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
echo 进入前端目录...
cd frontend

echo.
echo 检查依赖包...
if not exist "node_modules" (
    echo 📦 安装依赖包...
    npm install
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
) else (
    echo ✅ 依赖包已存在
)

echo.
echo 🚀 启动前端开发服务器...
echo 前端地址: http://localhost:3000
echo 按 Ctrl+C 停止服务器
echo.

npm run dev

pause
