"""
Preprocessing utilities for DocuMind RAG.

This module contains functions for:
- PDF text extraction
- text cleaning
- normalization
- sentence splitting
- sentence-based chunking
"""

import re
import fitz


def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extracts machine-readable text from uploaded PDF bytes.

    OCR is not used. This function only extracts selectable text.
    """

    text = ""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            text += f"\\n\\n--- Page {page_number} ---\\n\\n"
            text += page_text

    return text


def clean_text(text):
    """
    Cleans extracted academic and technical document text.

    The cleaning rules are intentionally general so that they work
    across different academic and technical PDFs.
    """

    # Remove page markers
    text = re.sub(r"--- Page \\d+ ---", " ", text)

    # Remove email addresses
    text = re.sub(r"\\S+@\\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", " ", text)

    # Remove common permission / attribution boilerplate if present
    text = re.sub(
        r"Provided proper attribution is provided,.*?scholarly works\\.",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove standalone page numbers
    text = re.sub(r"\\n\\s*\\d+\\s*\\n", "\\n", text)

    # Remove excessive newlines and tabs
    text = re.sub(r"\\n+", " ", text)
    text = re.sub(r"\\t+", " ", text)

    # Remove front matter before Abstract if Abstract appears early
    abstract_match = re.search(r"\\bAbstract\\b", text, flags=re.IGNORECASE)

    if abstract_match and abstract_match.start() < 5000:
        text = text[abstract_match.start():]

    # Remove references or bibliography section if it appears later
    reference_patterns = [
        r"\\bReferences\\b",
        r"\\bBibliography\\b"
    ]

    cut_positions = []

    for pattern in reference_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        for match in matches:
            if match.start() > len(text) * 0.5:
                cut_positions.append(match.start())

    if cut_positions:
        text = text[:min(cut_positions)]

    # Standardise spacing
    text = re.sub(r"\\s+", " ", text)

    return text.strip()


def split_into_sentences(text):
    """
    Lightweight sentence splitter for the Streamlit app.

    This avoids requiring NLTK at app runtime.
    """

    sentences = re.split(r"(?<=[.!?])\\s+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) > 3
    ]

    return sentences


def create_sentence_chunks(text, file_name, chunk_size=5, overlap=1):
    """
    Creates sentence-based chunks with overlap.

    Each chunk contains a group of sentences, not just one sentence.
    """

    sentences = split_into_sentences(text)

    chunks = []
    start = 0
    chunk_number = 1

    step = max(chunk_size - overlap, 1)

    while start < len(sentences):
        end = start + chunk_size
        chunk_sentences = sentences[start:end]

        chunk_text = " ".join(chunk_sentences).strip()

        if chunk_text:
            chunks.append({
                "Chunk ID": f"{file_name}_chunk_{chunk_number}",
                "File Name": file_name,
                "Chunk Number": chunk_number,
                "Chunk Text": chunk_text,
                "Word Count": len(chunk_text.split())
            })

            chunk_number += 1

        start += step

    return chunks


def normalize_text(text):
    """
    Normalizes text for TF-IDF retrieval.

    Decimal numbers such as 28.4, 41.8, and 3.5 are preserved.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", " ", text)

    # Protect decimal points inside numbers
    text = re.sub(r"(?<=\\d)\\.(?=\\d)", "decimalpointtoken", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\\s]", " ", text)

    # Restore decimal points
    text = text.replace("decimalpointtoken", ".")

    # Remove extra spaces
    text = re.sub(r"\\s+", " ", text).strip()

    return text