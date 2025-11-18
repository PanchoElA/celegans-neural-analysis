@echo off
REM open_interactive_graphs.bat
REM Open the main interactive dashboard (Neural_Dashboard_Interactive.html) in the default browser.

SETLOCAL
SET BASEDIR=%~dp0
IF NOT EXIST "%BASEDIR%interactive_code\Neural_Dashboard_Interactive.html" (
  echo ERROR: main dashboard not found: "%BASEDIR%interactive_code\Neural_Dashboard_Interactive.html"
  echo If you have the HTML under a different name, update this script or open it manually.
  pause
  ENDLOCAL
  EXIT /B 1
)

echo Opening main interactive dashboard...
start "" "%BASEDIR%interactive_code\Neural_Dashboard_Interactive.html"

ENDLOCAL
EXIT /B 0
