# rag_core/retrieval.py
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

def faiss_search(index, meta, query_vec, top_k=6) -> List[Dict]:
    D, I = index.search(query_vec, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        item = meta[idx].copy()
        item["score"] = float(score)
        hits.append(item)
    return hits

def bm25_rerank(query: str, candidates: List[Dict], k=4) -> List[Dict]:
    # lightweight hybrid: rerank with BM25 on text only
    docs = [c["text"] for c in candidates]
    bm = BM25Okapi([d.split() for d in docs])
    scores = bm.get_scores(query.split())
    for s, c in zip(scores, candidates):
        c["bm25"] = float(s)
        c["hybrid"] = 0.5*c["score"] + 0.5*(c["bm25"]/max(scores+[1]))
    return sorted(candidates, key=lambda x: x["hybrid"], reverse=True)[:k]
