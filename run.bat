@echo off
REM Convenience script to start the Ruangthong RAG service in dev mode (Windows).
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"

if not exist ".venv" (
    echo ==^> Creating virtualenv ^(.venv^)
    where py >nul 2>&1
    if %ERRORLEVEL%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Failed to create virtualenv. Make sure Python 3.10+ is installed.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtualenv.
    exit /b 1
)

echo ==^> Installing dependencies
python -m pip install --upgrade pip >nul
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

if not exist ".env" (
    echo ==^> Creating .env from .env.example ^(fill it in!^)
    copy /Y ".env.example" ".env" >nul
)

if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000

echo ==^> Starting FastAPI on http://localhost:%PORT%
uvicorn app.main:app --host %HOST% --port %PORT% --reload
