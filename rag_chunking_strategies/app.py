# app.py
import streamlit as st
from pathlib import Path
from engine.load_and_split import load_pdf, get_chunks
from utils.explain import get_strategy_explanation

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="RAG Chunking Explorer", layout="wide")
st.title("📄 RAG Chunking Strategy Explorer")
st.markdown("Upload a PDF and experiment with different chunking strategies used in Retrieval-Augmented Generation (RAG) systems.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("⚙️ Configuration")

pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, step=100, value=1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, step=50, value=200)

strategy = st.sidebar.selectbox("Chunking Strategy", [
    "Recursive (Fixed Size)",
    "Markdown-aware",
    "Semantic"
])

# ----------------------------
# Main Logic
# ----------------------------
if pdf_file:
    st.success(f"Uploaded: {pdf_file.name}")

    with st.spinner("📄 Loading and processing PDF..."):
        # Save uploaded file
        pdf_path = Path("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Extract raw text
        raw_text = load_pdf(pdf_path)

        # Strategy mapping
        if strategy == "Recursive (Fixed Size)":
            selected_strategy = "Recursive"
        elif strategy == "Markdown-aware":
            selected_strategy = "Markdown-aware"
        elif strategy == "Semantic":
            selected_strategy = "Semantic"
        else:
            selected_strategy = "Recursive"  # Fallback/default


        # Get chunks
        chunks = get_chunks(
            text=raw_text,
            strategy=selected_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    st.success(f"✅ Chunked into {len(chunks)} pieces using **{strategy}**")

    # ----------------------------
    # Strategy Explanation
    # ----------------------------
    st.subheader("📘 Strategy Explanation")
    st.markdown(get_strategy_explanation(selected_strategy))

    # ----------------------------
    # Chunk Visualization
    # ----------------------------
    st.subheader("🔍 Chunk Preview")
    for i, chunk in enumerate(chunks[:10]):
        with st.expander(f"Chunk {i+1} — {len(chunk)} chars"):
            st.markdown(chunk[:500] + "..." if len(chunk) > 500 else chunk)
