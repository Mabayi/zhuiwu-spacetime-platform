@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 算法环境缺失，请先配置项目 Python 环境。
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8080/#scene/slope"
".venv\Scripts\python.exe" server.py
pause
