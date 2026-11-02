@echo off
rem FreeAnimate Studio - one-time local install (no cloud services).
setlocal
cd /d "%~dp0"
title FreeAnimate Studio - Install

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ was not found on PATH.
  echo Install it from https://www.python.org/downloads/ and enable "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

set "PY=%~dp0.venv\Scripts\python.exe"
echo Installing core packages ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Review the output above.
  pause
  exit /b 1
)

echo.
echo Running health checks ...
"%PY%" -m freeanimate.cli doctor
"%PY%" -m freeanimate.cli hardware
echo.
echo Install complete. Run start.bat to open the studio.
pause
endlocal
