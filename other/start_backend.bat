@echo off
echo ========================================
echo   AI 数据分析助手 - 后端启动脚本
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
echo 检查依赖包...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo 📦 安装依赖包...
    pip install -r requirements.txt
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
echo 激活虚拟环境...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
) else (
    echo ⚠️  未找到虚拟环境，使用全局 Python
)

echo.
echo 🚀 启动后端 API 服务器...
echo 后端地址: http://localhost:8000
echo 按 Ctrl+C 停止服务器
echo.

python other/backend_api.py

pause
