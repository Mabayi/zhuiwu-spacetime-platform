@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo 请先安装 Python 3.12，然后重新运行本脚本。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" generate_sample.py
if errorlevel 1 goto failed
echo 安装完成，请双击启动演示.cmd。
pause
exit /b 0
:failed
echo 安装失败，请保留以上错误信息并检查网络或 Python 环境。
pause
exit /b 1
