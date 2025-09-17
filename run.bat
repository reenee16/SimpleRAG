@echo off
setlocal

REM Create venv if missing
if not exist venv (
  python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Start the app
streamlit run simple_ui.py

endlocal