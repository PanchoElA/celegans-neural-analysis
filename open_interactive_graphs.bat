@echo off
REM open_interactive_graphs.bat
REM Opens all .html files in the interactive_code folder with the system default browser.

SETLOCAL
SET BASEDIR=%~dp0
IF NOT EXIST "%BASEDIR%interactive_code" (
  echo ERROR: interactive_code folder not found: "%BASEDIR%interactive_code"
  pause
  ENDLOCAL
  EXIT /B 1
)

necho Opening interactive dashboards from "%BASEDIR%interactive_code"...
for %%f in ("%BASEDIR%interactive_code\*.html") do start "" "%%f"

nENDLOCAL
EXIT /B 0
