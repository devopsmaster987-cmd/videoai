@echo off
rem Stop the local FreeAnimate Studio server listening on PORT (default 8000).
setlocal
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=%PORT%; $c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { try { Stop-Process -Id $_ -Force; Write-Host ('Stopped process ' + $_) } catch { Write-Host ('Could not stop process ' + $_) } } } else { Write-Host ('No server listening on port ' + $p) }"
endlocal
