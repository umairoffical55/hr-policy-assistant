import os
import re
from typing import List, Dict

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# PDF + RAG functions
# -----------------------------
def extract_pdf_text(uploaded_file) -> List[Dict]:
    """Extract text page-by-page from an uploaded PDF."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    doc.close()
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping word-based chunks."""
    words = clean_text(text).split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    """Chunk the PDF while preserving page numbers."""
    all_chunks = []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "page": page_data["page"],
                    "chunk": chunk_index,
                    "text": chunk,
                }
            )

    return all_chunks


def build_faiss_index(chunks: List[Dict], model):
    """Embed chunks and build a cosine-similarity FAISS index."""
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(question: str, index, chunks: List[Dict], model, top_k: int = TOP_K):
    """Retrieve the most relevant chunks for a question."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def generate_answer(question: str, retrieved_chunks: List[Dict], client):
    """Ask GPT-OSS 20B to answer strictly from retrieved policy context."""
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}, chunk {item['chunk']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context provided below.

Rules:
1. Do not invent or assume policy details.
2. If the answer is not supported by the context, say:
   "I couldn't find that information in the uploaded HR policy."
3. Give a clear, practical answer.
4. When possible, mention the relevant policy page number(s).
5. If the policy contains conditions, exceptions, limits, or approval requirements, include them.
6. Do not provide legal advice. If a question requires legal interpretation, recommend contacting the appropriate HR/legal team.
"""

    user_prompt = f"""HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Answer based only on the policy context."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
        reasoning_effort="low",
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# UI
# -----------------------------
st.title("📘 HR Policy Assistant")
st.caption(
    "Upload an HR policy PDF and ask questions. "
    "The assistant retrieves relevant policy sections before answering."
)

with st.sidebar:
    st.header("1. Upload HR Policy")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload one HR policy document at a time.",
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing the HR policy..."):
                try:
                    pages = extract_pdf_text(uploaded_file)

                    if not pages:
                        st.error(
                            "No selectable text was found in this PDF. "
                            "If it is a scanned PDF, OCR is required."
                        )
                    else:
                        chunks = build_chunks(pages)
                        model = load_embedding_model()
                        index = build_faiss_index(chunks, model)

                        st.session_state.document_name = uploaded_file.name
                        st.session_state.pages = pages
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.messages = []

                        st.success("PDF processed successfully.")

                except Exception as exc:
                    st.error(f"Could not process the PDF: {exc}")

    st.divider()

    st.header("2. Knowledge Base")
    if st.session_state.document_name:
        st.write(f"**Document:** {st.session_state.document_name}")
        st.write(f"**Pages:** {len(st.session_state.pages)}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")
    else:
        st.info("Upload and process an HR policy PDF to begin.")

    st.divider()

    st.header("About")
    st.write(
        "RAG pipeline: PyMuPDF → text chunks → "
        "Sentence Transformers → FAISS → Groq GPT-OSS 20B."
    )

    if st.button("Clear document", use_container_width=True):
        st.session_state.document_name = None
        st.session_state.pages = []
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.messages = []
        st.rerun()


if not st.session_state.document_name:
    st.info("👈 Upload an HR policy PDF from the sidebar, then click **Process PDF**.")

    st.markdown(
        """
### Example questions

- What is the annual leave policy?
- How many sick leave days are allowed?
- What is the maternity leave entitlement?
- Who approves remote work?
- What is the probation period?
- What is the resignation notice period?
- What are the working hours?
"""
    )

else:
    st.success(
        f"Ready to answer questions from **{st.session_state.document_name}**."
    )

    # Show previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Retrieved policy sources"):
                    for source in message["sources"]:
                        st.write(
                            f"Page {source['page']} · "
                            f"Similarity {source['score']:.3f}"
                        )
                        st.caption(source["text"])

    question = st.chat_input("Ask a question about the HR policy...")

    if question:
        client = get_groq_client()

        if client is None:
            st.error(
                "GROQ_API_KEY is not configured. "
                "Add it in Streamlit Cloud → App settings → Secrets."
            )
            st.stop()

        model = load_embedding_model()

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching the policy..."):
                retrieved = retrieve(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    model,
                    top_k=TOP_K,
                )

            try:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, retrieved, client)

                st.markdown(answer)

                with st.expander("Retrieved policy sources"):
                    for source in retrieved:
                        st.write(
                            f"**Page {source['page']}** · "
                            f"Similarity: `{source['score']:.3f}`"
                        )
                        st.caption(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                    }
                )

            except Exception as exc:
                error_message = (
                    "I couldn't generate an answer because the Groq API "
                    f"returned an error: {exc}"
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
