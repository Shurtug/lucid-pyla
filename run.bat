@echo off
REM ============================================================
REM  PylaAI (Apex build) launcher
REM  Runs the improved bot from source. First run creates a
REM  Python 3.11 virtual environment (matching the original
REM  build) and installs the pinned dependencies; later runs
REM  just launch it.
REM ============================================================
setlocal
cd /d "%~dp0"

set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
    where uv >nul 2>&1
    if %errorlevel%==0 (
        echo [setup] Creating Python 3.11 venv via uv...
        uv venv "%VENV%" --python 3.11
        echo [setup] Installing pinned dependencies ^(first run only^)...
        uv pip install -r requirements.txt --python "%VENV%\Scripts\python.exe"
    ) else (
        echo [setup] uv not found; using py launcher ^(needs Python 3.11 installed^)...
        py -3.11 -m venv "%VENV%"
        if errorlevel 1 python -m venv "%VENV%"
        call "%VENV%\Scripts\activate.bat"
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo [setup] Dependency installation failed. See errors above.
        pause
        exit /b 1
    )
)

call "%VENV%\Scripts\activate.bat"
echo [run] Starting PylaAI ^(Apex^)...
python main.py
echo.
echo [run] Bot exited.
pause
