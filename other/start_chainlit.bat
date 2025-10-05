@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════╗
echo ║   🚀 AI 数据分析助手 - Chainlit 版本        ║
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

echo ✅ Python 环境正常
echo.

REM 检查 chainlit
pip show chainlit >nul 2>&1
if errorlevel 1 (
    echo 📦 安装 Chainlit...
    pip install chainlit>=1.0.0
    if errorlevel 1 (
        echo ❌ Chainlit 安装失败
        pause
        exit /b 1
    )
)

echo ✅ Chainlit 已安装
echo.

REM 检查其他依赖
echo 📦 检查依赖包...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ⚠️  部分依赖安装失败，但会尝试启动
)

echo.
echo ═══════════════════════════════════════════════
echo   🚀 启动 Chainlit 应用
echo ═══════════════════════════════════════════════
echo.
echo 📊 应用地址: http://localhost:8000
echo.
echo 💡 使用说明:
echo    1. 浏览器会自动打开应用
echo    2. 上传 CSV 文件
echo    3. 用自然语言提问开始分析
echo.
echo ⚠️  按 Ctrl+C 停止服务器
echo.

chainlit run chainlit_app.py -w

pause

