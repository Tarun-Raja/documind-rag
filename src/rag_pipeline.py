"""
RAG utilities for DocuMind RAG.

This follows the same RAG logic used in the notebook:
retrieved chunks → context → Gemini API answer generation.
"""

from google import genai


def build_rag_context(retrieved_chunks, max_chars_per_chunk=1000):
    """
    Builds context string from retrieved chunks.
    """

    context_parts = []

    for _, row in retrieved_chunks.iterrows():
        source_info = (
            f"[Source: {row['File Name']}, "
            f"Chunk: {row['Chunk Number']}, "
            f"Method: {row['Retrieval Method']}, "
            f"Score: {row['Similarity Score']:.4f}]"
        )

        chunk_text = str(row["Chunk Text"])[:max_chars_per_chunk]

        context_parts.append(source_info + "\n" + chunk_text)

    return "\n\n".join(context_parts)


def generate_answer_with_gemini(
    question,
    context,
    api_key,
    model_name="gemini-2.5-flash"
):
    """
    Generates a grounded answer using Gemini API.
    """

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an academic document question-answering assistant.

Answer the question using only the provided context.
Do not use outside knowledge.

If the answer is not available in the context, say:
"The document context does not provide enough information."

Keep the answer clear, concise, and academic.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text