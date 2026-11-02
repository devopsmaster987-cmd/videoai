@echo off
rem FreeAnimate Studio - check FFmpeg, Python, voices, hardware and capabilities.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo === Installation ===
"%PY%" -m freeanimate.cli doctor
echo.
echo === Hardware and capabilities ===
"%PY%" -m freeanimate.cli hardware
echo.
echo === Local voices (en) ===
"%PY%" -m freeanimate.cli providers
echo.
pause
endlocal
