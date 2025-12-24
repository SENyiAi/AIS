@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo          AIS - 依赖安装脚本 (Lite版本)
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未检测到Python，请先安装 Python 3.10 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [提示] 检测到Python环境
python --version
echo.

REM 询问是否创建虚拟环境
set /p CREATE_VENV="是否创建虚拟环境? (推荐) [Y/n]: "
if /i "%CREATE_VENV%"=="n" goto :INSTALL_DEPS
if /i "%CREATE_VENV%"=="N" goto :INSTALL_DEPS

:CREATE_VENV
echo.
echo [创建] 正在创建虚拟环境到 venv 目录...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo [错误] 虚拟环境创建失败
    pause
    exit /b 1
)

echo [完成] 虚拟环境创建成功
echo [激活] 正在激活虚拟环境...
call venv\Scripts\activate.bat
echo.

:INSTALL_DEPS
REM 询问是否使用清华源
set /p USE_MIRROR="是否使用清华源镜像加速安装? (推荐国内用户) [Y/n]: "
if /i "%USE_MIRROR%"=="n" goto :INSTALL_DEFAULT
if /i "%USE_MIRROR%"=="N" goto :INSTALL_DEFAULT

:INSTALL_MIRROR
echo.
echo [安装] 使用清华源安装依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
goto :CHECK_RESULT

:INSTALL_DEFAULT
echo.
echo [安装] 使用官方源安装依赖...
pip install -r requirements.txt

:CHECK_RESULT
if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 依赖安装失败
    echo.
    echo 可能的解决方案:
    echo   1. 检查网络连接
    echo   2. 尝试使用清华源: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo   3. 手动安装: pip install gradio
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [完成] 依赖安装成功！
echo.
if exist "venv\Scripts\activate.bat" (
    echo 使用方法:
    echo   1. 双击 启动.bat 运行程序
    echo   2. 或手动执行: venv\Scripts\activate.bat 激活环境后运行 python AIS_WebUI.py
) else (
    echo 使用方法:
    echo   双击 启动.bat 运行程序
)
echo ============================================================
pause
