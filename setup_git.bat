@echo off
echo Setting up Git repository for C. elegans Neural Analysis...
echo.

REM Try to find Git installation
set "GITPATH="
if exist "C:\Program Files\Git\bin\git.exe" set "GITPATH=C:\Program Files\Git\bin\git.exe"
if exist "C:\Program Files (x86)\Git\bin\git.exe" set "GITPATH=C:\Program Files (x86)\Git\bin\git.exe"

if "%GITPATH%"=="" (
    echo Git not found in standard locations.
    echo Please either:
    echo 1. Restart this terminal and try again
    echo 2. Install Git from: https://git-scm.com/download/win
    echo 3. Use GitHub Desktop instead
    pause
    exit /b 1
)

echo Found Git at: %GITPATH%
echo.

REM Initialize git repository
"%GITPATH%" init

REM Configure git if not already done
"%GITPATH%" config user.name >nul 2>&1
if errorlevel 1 (
    set /p username="Enter your name for Git: "
    "%GITPATH%" config user.name "!username!"
)

"%GITPATH%" config user.email >nul 2>&1
if errorlevel 1 (
    set /p email="Enter your email for Git: "
    "%GITPATH%" config user.email "!email!"
)

REM Add all files except those in .gitignore
"%GITPATH%" add .

REM Make initial commit
"%GITPATH%" commit -m "Initial commit: C. elegans neural activity analysis scripts - Added raster plot generation - Added PCA analysis - Added tuning curves analysis - Added basic time series visualization - Added Jupyter notebook for interactive analysis - Added comprehensive documentation"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub.com
echo 2. Copy the repository URL
echo 3. Run these commands (copy and paste each line):
echo.
echo "%GITPATH%" remote add origin [YOUR_GITHUB_REPO_URL]
echo "%GITPATH%" branch -M main
echo "%GITPATH%" push -u origin main
echo.
echo Example:
echo "%GITPATH%" remote add origin https://github.com/yourusername/celegans-neural-analysis.git
echo "%GITPATH%" branch -M main
echo "%GITPATH%" push -u origin main
pause