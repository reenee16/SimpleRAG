@echo off
setlocal

REM Check if Ollama API is responding
echo Checking if Ollama API is accessible...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama API is accessible. Proceeding with setup...
) else (
    echo Ollama API is not accessible. Checking if curl is available...
    curl --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: curl is not available. Please install curl or start Ollama manually.
        echo You can start Ollama by running: ollama serve
        pause
        exit /b 1
    )
    
    echo Starting Ollama server...
    start "" ollama serve
    
    REM Wait for Ollama to start up
    echo Waiting for Ollama to start...
    timeout /t 3 /nobreak >nul
    
    REM Check again if API is accessible (retry up to 10 times)
    setlocal enabledelayedexpansion
    set retries=0
    :check_ollama
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if !errorlevel! equ 0 (
        echo Ollama is now running and accessible.
        goto ollama_ready
    )
    
    set /a retries+=1
    if !retries! lss 2 (
        echo Waiting for Ollama to be ready... (attempt !retries!/10^)
        timeout /t 2 /nobreak >nul
        goto check_ollama
    ) else (
        echo Error: Ollama failed to start or become accessible after 10 attempts.
        echo Please check if Ollama is properly installed and try starting it manually.
        pause
        exit /b 1
    )
)

:ollama_ready

REM Create venv if missing
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip and install deps
echo Upgrading pip...
python -m pip install --upgrade pip

@REM echo Installing dependencies...
pip install -r requirements.txt

REM Start the app
echo Starting Streamlit application...
streamlit run simple_ui.py

endlocal