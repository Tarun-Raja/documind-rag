"""
Retrieval utilities for DocuMind RAG.

This follows the same retrieval logic used in the notebook:
- TF-IDF over preprocessed chunks
- embedding retrieval over cleaned readable chunks
- hybrid retrieval combining both
"""

import numpy as np
import pandas as pd
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_for_tfidf


def build_tfidf_index(chunk_texts):
    """
    Builds a TF-IDF vectorizer and TF-IDF matrix using the same
    preprocessing process used in the notebook.
    """

    processed_chunks = [
        preprocess_for_tfidf(text)
        for text in chunk_texts
    ]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(processed_chunks)

    return vectorizer, tfidf_matrix


def build_faiss_index(chunk_texts, embedding_model):
    """
    Builds a FAISS index from cleaned readable chunk embeddings.
    """

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_tfidf(query, chunks_df, vectorizer, tfidf_matrix, top_k=3):
    """
    Retrieves top-k chunks using TF-IDF and cosine similarity.
    """

    top_k = min(top_k, len(chunks_df))

    processed_query = preprocess_for_tfidf(query)

    query_vector = vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    results = chunks_df.iloc[top_indices].copy()
    results["Similarity Score"] = similarity_scores[top_indices]
    results["Retrieval Method"] = "TF-IDF"

    return results


def retrieve_embeddings(query, chunks_df, faiss_index, embedding_model, top_k=3):
    """
    Retrieves top-k chunks using sentence embeddings and FAISS.
    """

    top_k = min(top_k, len(chunks_df))

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = chunks_df.iloc[indices[0]].copy()
    results["Similarity Score"] = scores[0]
    results["Retrieval Method"] = "Embedding"

    return results


def retrieve_hybrid(
    query,
    chunks_df,
    vectorizer,
    tfidf_matrix,
    faiss_index,
    embedding_model,
    top_k_tfidf=3,
    top_k_embedding=3
):
    """
    Combines TF-IDF retrieval and embedding retrieval.

    This matches the notebook's hybrid retrieval approach.
    """

    tfidf_results = retrieve_tfidf(
        query=query,
        chunks_df=chunks_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k_tfidf
    )

    embedding_results = retrieve_embeddings(
        query=query,
        chunks_df=chunks_df,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        top_k=top_k_embedding
    )

    combined_results = pd.concat(
        [tfidf_results, embedding_results],
        ignore_index=True
    )

    combined_results = combined_results.drop_duplicates(
        subset=["Chunk ID"],
        keep="first"
    )

    combined_results = combined_results.reset_index(drop=True)

    return combined_results