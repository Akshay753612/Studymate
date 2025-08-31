# rag_core/ingest.py
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict

def extract_chunks(pdf_path: Path, max_chars=1200, overlap=150) -> List[Dict]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # rough normalize
        text = " ".join(text.split())
        # simple sliding window chunking
        i = 0
        while i < len(text):
            chunk = text[i:i+max_chars]
            chunks.append({
                "doc_id": pdf_path.name,
                "page": page_num,
                "text": chunk,
            })
            i += max_chars - overlap
    doc.close()
    return chunks

def ingest_folder(folder: Path) -> List[Dict]:
    all_chunks = []
    for pdf in folder.glob("*.pdf"):
        all_chunks.extend(extract_chunks(pdf))
    return all_chunks
