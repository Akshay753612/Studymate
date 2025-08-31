# app.py — StudyMate (Multi-PDF Q&A + Chat with Gemini 2.5 Flash)
import os, io
import numpy as np
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from google import genai

# ---------- Look & Feel ----------
st.set_page_config(page_title="StudyMate – PDF Q&A", layout="wide")
PRIMARY = "#947BAE"  # violet-500 accent
BG = "#4EB2E0"       # dark background
CARD = "#FCF7F7"     # panel color
TEXT = "#121111E0"     # slate-200
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {BG};
        color: {TEXT};
      }}
      .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 2rem;
      }}
      h1, h2, h3, h4 {{
        color: {TEXT};
      }}
      .stButton>button {{
        background:{PRIMARY}; color:white; border:0; padding:0.6rem 1rem;
        border-radius:10px; font-weight:600;
      }}
      .stTextInput>div>div>input, .stTextArea textarea {{
        background:{CARD} !important; color:{TEXT} !important; border-radius:10px;
      }}
      .uploadedFile {{"background": "{CARD} !important"}}
      .stFileUploader > div > div {{"background": "{CARD} !important"}}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"# 🧠 StudyMate")
st.caption(" an AI powered PDF-based Q&A system for students")
API_KEY = "AIzaSyCaTi3RV-LPnv6l83QdUKsfENIKsGWKHqA"  # <<< paste your key here
client = genai.Client(api_key=API_KEY)
def extract_pdf_text(file_obj: io.BytesIO) -> str:
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    pages = [p.get_text() for p in doc]
    doc.close()
    return "\n".join(pages)
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
def build_index(all_chunks):
    """all_chunks: list[dict] with keys: text, source"""
    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx, embs.shape[1]

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    k = st.slider("Chunks to retrieve", 1, 12, 4)
    show_context = st.checkbox("Show retrieved context", value=True)
st.sidebar.info(
    "This application allows you to upload a PDFs document and then allows you to ask questions about its content. "
    "It also includes a general-purpose chatbot for easy use."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤ using Streamlit, FAISS, PyMuPDF, NumPy, Sentence Transformers and Google Gemini  ")    

# ---------- Multi-PDF upload ----------
st.subheader("📄 Upload PDFs")
files = st.file_uploader(
    "Upload one or more PDFs", type="pdf", accept_multiple_files=True
)

if files:
    # Extract and chunk per file
    all_chunks, file_map = [], []  # file_map aligns with all_chunks
    for f in files:
        with st.spinner(f"Extracting: {f.name}"):
            text = extract_pdf_text(f)
        chunks = chunk_text(text, 500)
        for ch in chunks:
            all_chunks.append({"text": ch, "source": f.name})
            file_map.append(f.name)

    # Build FAISS
    with st.spinner("Building embeddings & index…"):
        index, dim = build_index(all_chunks)
        embedder = get_embedder()

    st.success(f"Loaded **{len(files)}** file(s), **{len(all_chunks)}** chunks.")

    # Filter which files to search
    file_names = sorted({c["source"] for c in all_chunks})
    which = st.multiselect(
        "Limit question to specific files (optional):",
        options=file_names,
        default=file_names,
    )

    # ---------- Ask a question ----------
    st.subheader("🔎 Ask a question about your PDFs")
    q = st.text_area("Type your question:", placeholder="e.g., Summarize chapter 1 across all files")

    colA, colB = st.columns([1,1])
    with colA:
        ask = st.button("Answer with StudyMate")
    with colB:
        summarize_all = st.button("Full summary of all uploaded PDFs")

    # Prepare mask for selected files
    sel_idx = [i for i, src in enumerate(file_map) if src in which]

    if ask and q:
        q_emb = embedder.encode([q], convert_to_numpy=True).astype("float32")
        # Restrict search to selected files
        D, I = index.search(q_emb, k=min(k, len(sel_idx)))
        # Re-rank to only include chunks from selected files
        ranked = [i for i in I[0] if i in sel_idx][:k]
        retrieved = [all_chunks[i] for i in ranked]
        context = "\n\n---\n\n".join([f"[{c['source']}]\n{c['text']}" for c in retrieved])

        prompt = (
            "You are StudyMate, a helpful academic assistant. "
            "Answer ONLY using the context below. If the answer is not present, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            "Answer:"
        )
        with st.spinner("StudyMate is thinking…"):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                st.markdown("### ✅ Answer")
                st.write(resp.text)
            except Exception as e:
                st.error(f"Gemini error: {e}")

        if show_context:
            with st.expander("🔍 Retrieved context"):
                st.write(context)

    # ---------- Full summary across all PDFs ----------
    if summarize_all:
        BIG_CONTEXT_LIMIT = 12000  # rough guard so prompt doesn't explode
        texts_joined = "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in all_chunks
        )
        if len(texts_joined) > BIG_CONTEXT_LIMIT:
            # Use top-N chunks by naive length (or you can sample)
            texts_joined = texts_joined[:BIG_CONTEXT_LIMIT]

        prompt = (
            "Create a concise, structured summary of the following combined PDF contents. "
            "Use headings and bullet points where helpful. Mention file names when relevant.\n\n"
            f"{texts_joined}\n\n"
            "Return the final summary only."
        )
        with st.spinner("Summarizing all PDFs…"):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                st.markdown("### 📚 Summary of all PDFs")
                st.write(resp.text)
            except Exception as e:
                st.error(f"Gemini error: {e}")
st.markdown("---")
st.subheader("💬 Quick Chat (StudyMate)")
chat_q = st.text_input("Ask anything:")
if chat_q:
    with st.spinner("Thinking…"):
        try:
            r = client.models.generate_content(model="gemini-2.5-flash", contents=chat_q)
            st.markdown("### 💡 Response")
            st.write(r.text)
        except Exception as e:
            st.error(f"Gemini error: {e}")
