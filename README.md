# DocuMind RAG

## Explainable NLP and Retrieval-Augmented QA for Academic and Technical Documents

DocuMind RAG is an explainable NLP and Retrieval-Augmented Question Answering system designed for academic and technical documents such as research papers, lecture notes, technical reports, manuals, and educational PDFs.

The project demonstrates a complete NLP pipeline before applying modern retrieval and generation methods.

## Project Objective

The goal of this project is to build a document question-answering system that is both practical and explainable.

Instead of directly applying a large language model, the project first demonstrates classical NLP steps such as text cleaning, tokenization, normalization, stopword removal, stemming, lemmatization, POS tagging, dependency parsing, and encoding techniques.

The final system compares TF-IDF retrieval, embedding-based retrieval, hybrid retrieval, and Gemini-based RAG answer generation.

## Key Features

- PDF text extraction using PyMuPDF
- Text cleaning for academic and technical documents
- Tokenization and normalization
- Stopword removal
- Stemming and lemmatization comparison
- POS tagging and dependency parsing
- Bag of Words and TF-IDF encoding
- Sentence embeddings using SentenceTransformers
- FAISS-based semantic retrieval
- Hybrid retrieval using TF-IDF and embeddings
- Gemini API-based RAG answer generation
- Retrieved source chunk display for explainability
- Streamlit app with PDF upload and question answering

## NLP Pipeline

The project demonstrates the following NLP concepts:

1. Text cleaning
2. Tokenization
3. Normalization
4. Stopword removal
5. Stemming and lemmatization
6. POS tagging
7. Dependency parsing
8. Encoding techniques:
   - Bag of Words
   - TF-IDF
   - Sentence embeddings

## RAG Architecture

The final RAG system follows this flow:

    Uploaded PDF Documents
            ↓
    Text Extraction
            ↓
    Text Cleaning
            ↓
    Sentence-Based Chunking
            ↓
    TF-IDF Retrieval + Embedding Retrieval
            ↓
    Hybrid Retrieved Context
            ↓
    Gemini API Answer Generation
            ↓
    Answer + Retrieved Source Chunks

## Folder Structure

    documind-rag/
    │
    ├── notebooks/
    │   └── DocuMind_RAG_Explainable_NLP_QA_Project.ipynb
    │
    ├── app/
    │   └── streamlit_app.py
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── retrieval.py
    │   └── rag_pipeline.py
    │
    ├── data/
    │   ├── raw_documents/
    │   └── processed_text/
    │
    ├── outputs/
    │   ├── retrieved_chunks/
    │   └── evaluation/
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Running the Notebook

Open the notebook in Google Colab:

    notebooks/DocuMind_RAG_Explainable_NLP_QA_Project.ipynb

Run all cells from top to bottom.

The notebook is the main academic deliverable and demonstrates the full NLP pipeline step by step.

## Running the Streamlit App

Install dependencies:

    pip install -r requirements.txt

Run the app:

    streamlit run app/streamlit_app.py

## Setting Up the Gemini API Key

The Streamlit app requires a Gemini API key for answer generation.

### Step 1: Get a Gemini API Key

1. Go to Google AI Studio:

   ```
   https://aistudio.google.com/app/apikey
   ```

2. Sign in with your Google account.

3. Click **Create API Key**.

4. Copy the generated API key.

### Step 2: Create the `.streamlit` Folder

Inside the project root folder, create a folder named:

```
.streamlit
```

Final structure:

```
documind-rag/
└── .streamlit/
```

### Step 3: Create `secrets.toml`

Inside the `.streamlit` folder, create a file named:

```
secrets.toml
```

### Step 4: Add the API Key

Open `secrets.toml` and add:

```
GEMINI_API_KEY = "your_actual_api_key_here"
```

Example:

```
GEMINI_API_KEY = "AIzaSyxxxxxxxxxxxxxxxx"
```
## spaCy Model Setup

The project uses the spaCy English language model:

```
en_core_web_sm
```

After installing the project requirements, run:

```
python -m spacy download en_core_web_sm
```

This downloads the language model required for lemmatization and other NLP tasks.

### Step 5: Run the Streamlit App

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/streamlit_app.py
```

The app will now securely access the Gemini API key through Streamlit secrets.

### Important

Do not upload `secrets.toml` to GitHub.

The `.gitignore` file already excludes it automatically.


## Modular Project Structure

The GitHub version separates reusable logic into the src folder.

- src/preprocessing.py contains PDF extraction, text cleaning, normalization, and chunking utilities.
- src/retrieval.py contains TF-IDF retrieval, embedding retrieval, FAISS indexing, and hybrid retrieval utilities.
- src/rag_pipeline.py contains RAG context construction and Gemini answer generation utilities.

The notebook remains the main academic explanation, while the Streamlit app imports reusable functions from src.

## Scope

This project focuses on machine-readable academic and technical documents.

It does not perform OCR. Scanned PDFs, handwritten notes, images, charts, and diagrams are outside the current scope.

## Results Summary

The project showed that:

- TF-IDF retrieval works well for exact keyword, acronym, and definition-based questions.
- Embedding retrieval works well for semantic and conceptual questions.
- Hybrid retrieval improves the final RAG system by combining exact keyword matching and semantic similarity.
- RAG answer generation becomes more explainable when retrieved source chunks are displayed with the answer.
- Classical NLP preprocessing remains important even when modern LLM-based systems are used.

## Limitations

- Small document collection in the notebook version
- Qualitative evaluation only
- Simple sentence-based chunking
- Gemini API required for answer generation
- No OCR support for scanned PDFs

## Future Improvements

- Add OCR support
- Add section-aware chunking
- Add reranking
- Add quantitative evaluation metrics
- Add citation-style answer formatting
- Improve Streamlit deployment
