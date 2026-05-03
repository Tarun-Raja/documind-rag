"""
Preprocessing utilities for DocuMind RAG.

This follows the same preprocessing logic used in the notebook:
- PDF text extraction
- text cleaning
- sentence tokenization
- normalization
- stopword removal
- spaCy lemmatization
- sentence-based chunking
"""

import re
import fitz
import nltk
import spacy

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def setup_nlp_resources():
    """
    Downloads and loads required NLP resources.
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return nlp


nlp = setup_nlp_resources()
stop_words = set(stopwords.words("english"))


def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extracts machine-readable text from uploaded PDF bytes.
    OCR is not used.
    """

    text = ""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            text += f"\n\n--- Page {page_number} ---\n\n"
            text += page_text

    return text


def clean_text(text):
    """
    Cleans raw extracted PDF text from academic and technical documents.
    """

    text = re.sub(r"--- Page \d+ ---", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(
        r"Provided proper attribution is provided,.*?scholarly works\.",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", " ", text)

    abstract_match = re.search(r"\bAbstract\b", text, flags=re.IGNORECASE)

    if abstract_match and abstract_match.start() < 5000:
        text = text[abstract_match.start():]

    reference_patterns = [
        r"\bReferences\b",
        r"\bBibliography\b"
    ]

    cut_positions = []

    for pattern in reference_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        for match in matches:
            if match.start() > len(text) * 0.5:
                cut_positions.append(match.start())

    if cut_positions:
        text = text[:min(cut_positions)]

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_text(text):
    """
    Normalizes text while preserving decimal values.
    """

    text = str(text).lower()

    text = re.sub(r"(?<=\d)\.(?=\d)", "decimalpointtoken", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = text.replace("decimalpointtoken", ".")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text):
    """
    Removes common English stopwords.
    """

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def lemmatize_text(text):
    """
    Applies spaCy lemmatization.
    """

    doc = nlp(text)

    lemmas = [
        token.lemma_
        for token in doc
        if token.lemma_.strip() != ""
    ]

    return " ".join(lemmas)


def preprocess_for_tfidf(text):
    """
    Applies the same TF-IDF preprocessing used in the notebook:
    normalization → stopword removal → lemmatization.
    """

    normalized = normalize_text(text)
    no_stopwords = remove_stopwords(normalized)
    lemmatized = lemmatize_text(no_stopwords)

    return lemmatized


def create_sentence_chunks(text, file_name, chunk_size=5, overlap=1):
    """
    Creates sentence-based chunks with overlap using NLTK sentence tokenization.
    """

    sentences = sent_tokenize(text)

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