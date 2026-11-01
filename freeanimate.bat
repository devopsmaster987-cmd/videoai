@echo off
rem FreeAnimate AI — Windows launcher
rem Usage: freeanimate.bat [command] [args...]
setlocal
set "SCRIPT_DIR=%~dp0"

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%SCRIPT_DIR%freeanimate" %*
) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        py -3 "%SCRIPT_DIR%freeanimate" %*
    ) else (
        echo Python was not found. Install Python 3.10+ from https://python.org
        exit /b 1
    )
)
endlocal
