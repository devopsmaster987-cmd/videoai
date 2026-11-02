@echo off
rem FreeAnimate Studio - start the local web UI and open it in a browser.
setlocal
cd /d "%~dp0"
title FreeAnimate Studio

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

set "PY=python"
if exist ".venv\Scripts\python.exe" (
  rem Use the venv only if it actually has the web dependencies.
  ".venv\Scripts\python.exe" -c "import fastapi, uvicorn, jinja2" >nul 2>&1
  if not errorlevel 1 set "PY=%~dp0.venv\Scripts\python.exe"
)

echo Starting FreeAnimate Studio at http://127.0.0.1:%PORT%/
echo Press Ctrl+C in this window to stop the server.
echo.
"%PY%" -m freeanimate.cli serve --port %PORT% --open
if errorlevel 1 (
  echo.
  echo FreeAnimate failed to start. The error is shown above.
  pause
)
endlocal
