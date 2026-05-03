import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer


# Make src importable when running from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.preprocessing import (
    extract_text_from_pdf_bytes,
    clean_text,
    create_sentence_chunks
)

from src.retrieval import (
    build_tfidf_index,
    build_faiss_index,
    retrieve_hybrid
)

from src.rag_pipeline import (
    build_rag_context,
    generate_answer_with_gemini
)


st.set_page_config(
    page_title="DocuMind RAG",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    """
    Loads the sentence embedding model.
    """

    return SentenceTransformer("all-MiniLM-L6-v2")


def get_gemini_api_key():
    """
    Gets Gemini API key from Streamlit secrets or environment variable.
    """

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


st.title("DocuMind RAG")
st.subheader("Explainable NLP and Retrieval-Augmented QA for Academic and Technical Documents")

st.markdown(
    """
    Upload academic or technical PDF documents and ask questions about their content.

    This app performs machine-readable PDF text extraction, text cleaning,
    sentence-based chunking, TF-IDF retrieval, embedding retrieval with FAISS,
    hybrid retrieval, Gemini-based answer generation, and source chunk display.

    OCR for scanned PDFs is not included.
    """
)


# Sidebar Settings

st.sidebar.header("Chunking Settings")

chunk_size = st.sidebar.slider(
    "Sentences per chunk",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

overlap = st.sidebar.slider(
    "Sentence overlap",
    min_value=0,
    max_value=3,
    value=1,
    step=1
)

st.sidebar.header("Retrieval Settings")

top_k_tfidf = st.sidebar.slider(
    "Number of TF-IDF chunks",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

top_k_embedding = st.sidebar.slider(
    "Number of embedding chunks",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

show_context = st.sidebar.checkbox(
    "Show full RAG context",
    value=False
)


# Upload and Processing

uploaded_files = st.file_uploader(
    "Upload academic or technical PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

process_button = st.button("Process Uploaded Documents")

embedding_model = load_embedding_model()

if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing uploaded documents..."):

            all_chunks = []
            document_summary = []

            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                pdf_bytes = uploaded_file.getvalue()

                raw_text = extract_text_from_pdf_bytes(pdf_bytes)
                cleaned_text = clean_text(raw_text)

                chunks = create_sentence_chunks(
                    text=cleaned_text,
                    file_name=file_name,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

                all_chunks.extend(chunks)

                document_summary.append({
                    "File Name": file_name,
                    "Extracted Characters": len(raw_text),
                    "Cleaned Characters": len(cleaned_text),
                    "Number of Chunks": len(chunks)
                })

            chunks_df = pd.DataFrame(all_chunks)

            if chunks_df.empty:
                st.error(
                    "No text chunks were created. The uploaded PDFs may be scanned or image-based."
                )
                st.stop()

            chunk_texts = chunks_df["Chunk Text"].tolist()

            tfidf_vectorizer, tfidf_matrix = build_tfidf_index(chunk_texts)

            faiss_index, chunk_embeddings = build_faiss_index(
                chunk_texts,
                embedding_model
            )

            st.session_state["chunks_df"] = chunks_df
            st.session_state["document_summary_df"] = pd.DataFrame(document_summary)
            st.session_state["tfidf_vectorizer"] = tfidf_vectorizer
            st.session_state["tfidf_matrix"] = tfidf_matrix
            st.session_state["faiss_index"] = faiss_index
            st.session_state["documents_processed"] = True

        st.success("Documents processed successfully.")


if st.session_state.get("documents_processed", False):
    st.subheader("Uploaded Document Summary")
    st.dataframe(st.session_state["document_summary_df"])

    st.subheader("Chunk Preview")
    st.dataframe(
        st.session_state["chunks_df"][
            ["Chunk ID", "File Name", "Chunk Number", "Word Count", "Chunk Text"]
        ].head(10)
    )


# Question Answering

st.subheader("Ask a Question")

example_questions = [
    "What is the main idea of the document?",
    "What method does the paper propose?",
    "What are the key findings?",
    "What are the limitations mentioned?",
    "How does the proposed system work?"
]

selected_example = st.selectbox(
    "Choose an example question or type your own below:",
    [""] + example_questions
)

question = st.text_area(
    "Enter your question:",
    value=selected_example,
    height=100,
    placeholder="Ask a question about the uploaded academic or technical documents..."
)

answer_button = st.button("Generate Answer")

if answer_button:
    if not st.session_state.get("documents_processed", False):
        st.warning("Please upload and process documents first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        api_key = get_gemini_api_key()

        if not api_key:
            st.error(
                "Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets "
                "or set it as an environment variable."
            )
            st.stop()

        with st.spinner("Retrieving relevant chunks and generating answer..."):

            chunks_df = st.session_state["chunks_df"]
            tfidf_vectorizer = st.session_state["tfidf_vectorizer"]
            tfidf_matrix = st.session_state["tfidf_matrix"]
            faiss_index = st.session_state["faiss_index"]

            retrieved_chunks = retrieve_hybrid(
                query=question,
                chunks_df=chunks_df,
                vectorizer=tfidf_vectorizer,
                tfidf_matrix=tfidf_matrix,
                faiss_index=faiss_index,
                embedding_model=embedding_model,
                top_k_tfidf=top_k_tfidf,
                top_k_embedding=top_k_embedding
            )

            rag_context = build_rag_context(
                retrieved_chunks,
                max_chars_per_chunk=1000
            )

            answer = generate_answer_with_gemini(
                question=question,
                context=rag_context,
                api_key=api_key
            )

        st.subheader("Generated Answer")
        st.write(answer)

        st.subheader("Retrieved Source Chunks")

        for i, row in retrieved_chunks.iterrows():
            with st.expander(
                f"Source {i + 1}: {row['File Name']} | "
                f"Chunk {row['Chunk Number']} | "
                f"{row['Retrieval Method']} | "
                f"Score: {row['Similarity Score']:.4f}"
            ):
                st.write(row["Chunk Text"])

        if show_context:
            st.subheader("Full RAG Context Sent to Gemini")
            st.text_area(
                "RAG Context",
                value=rag_context,
                height=400
            )