@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo          AIS - AI Image Super-Resolution
echo                     By SENyiAi
echo ============================================================
echo.

REM 优先检查虚拟环境 (Lite版本)
if exist "venv\Scripts\python.exe" (
    echo [检测] 使用虚拟环境 Python
    set "PYTHON_EXE=venv\Scripts\python.exe"
    goto :RUN
)

REM 检查是否有内置Python (Full版本)
if exist "prereq\python-3.12.10-embed-amd64\python.exe" (
    echo [检测] 使用内置 Python 3.12 环境
    set "PYTHON_EXE=prereq\python-3.12.10-embed-amd64\python.exe"
    goto :RUN
)

REM 使用系统Python
echo [检测] 使用系统 Python 环境
set "PYTHON_EXE=python"

:RUN
echo [启动] 正在启动 WebUI...
echo.

"%PYTHON_EXE%" AIS_WebUI.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误代码: %ERRORLEVEL%
    echo.
    echo 可能的解决方案:
    echo   1. 确保已安装 Python 3.10+
    echo   2. Lite版本请先运行 "安装依赖.bat"
    echo   3. 检查是否安装了 gradio: pip install gradio
    echo   2. 运行: pip install -r requirements.txt
    echo   3. 确保模型文件夹中有超分引擎
    echo.
)

pause
