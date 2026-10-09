@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please create the environment first. See README.md.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:18765 after the server is ready.
".venv\Scripts\python.exe" demo_server.py
pause
