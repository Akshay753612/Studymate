# rag_core/generator.py
import os
from typing import List, Dict

# Example: IBM watsonx.ai
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

def _watsonx_model():
    creds = Credentials(
        url=os.environ["https://eu-de.ml.cloud.ibm.com"],
        api_key=os.environ["kQUUvFZRMmKZym5h-kMgq-NTjPxFKEdKZdXNXZmmikAd"]
    )
    project_id = os.environ["bc84e479-50d8-4038-99e4-b1f848c9a511"]
    # Mistral 8x7B Instruct id may vary; update to your deployed model id
    model_id = os.getenv("WATSONX_MODEL_ID", "mistral-7b-instruct-v0.2")
    return Model(model_id=model_id, credentials=creds, project_id=project_id)

SYS_PROMPT = """You are StudyMate, a helpful assistant that answers strictly
from the provided context. If the answer is not present, say you don't know.
Always cite sources as (DocName p.PageNum)."""

def format_context(chunks: List[Dict], max_chars=4000):
    ctx = ""
    citations = []
    for i, c in enumerate(chunks, 1):
        tag = f"[{i}] ({c['doc_id']} p.{c['page']})"
        citations.append(tag)
        ctx_piece = f"{tag}\n{c['text']}\n\n"
        if len(ctx + ctx_piece) > max_chars: break
        ctx += ctx_piece
    return ctx, citations

def answer_with_llm(question: str, chunks: List[Dict]) -> str:
    ctx, citations = format_context(chunks)
    prompt = f"{SYS_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    model = _watsonx_model()
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 350,
        GenParams.TEMPERATURE: 0.0,
    }
    resp = model.generate_text(prompt=prompt, params=params)
    text = resp.get("results", [{}])[0].get("generated_text", "").strip()
    # Ensure citations present (fallback)
    if "(" not in text:
        text += "\n\nSources: " + ", ".join(citations)
    return text
