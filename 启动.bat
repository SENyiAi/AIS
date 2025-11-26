@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo          AIS - AI Image Super-Resolution
echo                     By SENyiAi
echo ============================================================
echo.

REM 检查是否有内置Python (优先检测3.14)
if exist "前置\python-3.14.0-embed-amd64\python.exe" (
    echo [检测] 使用内置 Python 3.14 环境
    set "PYTHON_EXE=前置\python-3.14.0-embed-amd64\python.exe"
) else (
    echo [检测] 使用系统 Python 环境
    set "PYTHON_EXE=python"
)

echo [启动] 正在启动 WebUI...
echo.

"%PYTHON_EXE%" AIS_WebUI.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误代码: %ERRORLEVEL%
    echo.
    echo 可能的解决方案:
    echo   1. 确保已安装 Python 3.10+
    echo   2. 运行: pip install -r requirements.txt
    echo   3. 确保模型文件夹中有超分引擎
    echo.
)

pause
