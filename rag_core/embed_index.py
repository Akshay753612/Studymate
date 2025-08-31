# rag_core/embed_index.py
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path

MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")

def build_index(chunks: List[Dict], index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(X)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta = [json.loads(line) for line in open(index_dir / "meta.jsonl", "r", encoding="utf-8")]
    return index, meta

def embed_query(query: str):
    model = SentenceTransformer(MODEL_NAME)
    v = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float32)
