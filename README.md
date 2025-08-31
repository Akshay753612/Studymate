# StudyMate

Multi-PDF Q&A and summarization app built with **Streamlit** + **Gemini 2.5 Flash**.

## Features
- Upload multiple PDFs (multi-file)
- Context-aware Q&A (SentenceTransformers + FAISS retrieval)
- Full-document summary across PDFs
- Dark themed UI

## Run locally
```bash
pip install -r requirements.txt
# Windows CMD
set GEMINI_API_KEY=YOUR_KEY
# PowerShell
$env:GEMINI_API_KEY="YOUR_KEY"
streamlit run app.py
