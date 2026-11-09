@echo off
REM One-time setup (Windows). Needs Python 3.11 on the PATH (python --version).
cd /d "%~dp0"
python --version
if errorlevel 1 (echo Python was not found. Install Python 3.11 from python.org and run this again. & pause & exit /b 1)
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 (
  echo.
  echo WARNING: this project was tested with Python 3.11 only. Other versions may fail to install the pinned packages.
  set /p CONT=Continue anyway? Type y to continue, anything else to stop: 
  if /i not "%CONT%"=="y" exit /b 1
)
echo.
echo [1/3] Creating the environment and installing packages (several minutes)...
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (echo Installation failed. See the message above and README section 3. & pause & exit /b 1)
echo.
echo [2/3] Google Earth Engine project
echo Enter the ID of your Google Cloud project that has the Earth Engine API enabled (see README section 2).
set /p EEP=Project ID: 
echo %EEP%> ee_project.txt
echo.
echo [3/3] Earth Engine login: a browser window opens. Sign in with your Google account and allow access.
.venv\Scripts\earthengine authenticate
echo.
echo Setup finished. Start the app with run_app.bat
pause
