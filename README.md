# PDF-AI

Upload up to 5 PDFs and ask questions with retrieval-augmented generation.

## Prerequisites
- Python 3.10+
- Ollama installed and running (`ollama serve`)
- Pull a small model (example):
  - `ollama pull llama3.2:3b`

## One-click run

Windows:
- Double-click `run.bat` (or run in Terminal: `run.bat`)

macOS/Linux:
- `chmod +x start.sh`
- `./start.sh`

The app will open at `http://localhost:8501`.

## Notes
- Change the default Ollama model from the sidebar in the app if needed.
- If PDFs fail to parse, install PyMuPDF: `pip install PyMuPDF` (already in requirements).
