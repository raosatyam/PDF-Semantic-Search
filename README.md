# PDF Semantic Search

This project implements a semantic search system for PDF documents using Sentence Transformers and FAISS for efficient similarity search.

## Features

* **PDF Text Extraction:** Extracts text from PDF files using `PyPDF2`.
* **Sentence Embeddings:** Generates sentence embeddings using Sentence Transformers.
* **Vector Indexing:** Uses FAISS for fast similarity search.
* **Simple Web Interface:** Provides a basic Flask-based web interface with upload and search functionality.

## Requirements

* Python 3.6+
* `PyPDF2`
* `sentence-transformers`
* `faiss-cpu` (or `faiss-gpu` for GPU support)
* `Flask`

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/raosatyam/PDF-Semantic-Search.git](https://www.google.com/search?q=https://github.com/raosatyam/PDF-Semantic-Search.git)
    cd PDF-Semantic-Search
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage and API Endpoints

1.  **Run the application:**

    ```bash
    python app.py
    ```

2.  **API Endpoints:**

    * **`/upload` (POST):**
        * This endpoint allows you to upload PDF files to the server.
        * **Request:**
            * The request should be a `multipart/form-data` request.
            * The PDF file should be sent as a file named `pdf_file`.
        * **Example using curl:**
            ```bash
            curl -X POST -F "pdf_file=@/path/to/your/document.pdf" [http://127.0.0.1:5000/upload](https://www.google.com/search?q=http://127.0.0.1:5000/upload)
            ```
        * **Response:**
            * A JSON response indicating whether the upload was successful.
            * Example success response: `{"message": "PDF uploaded successfully"}`
            * Example fail response: `{"error": "Failed to upload PDF"}`

    * **`/search` (POST):**
        * This endpoint performs a semantic search on the uploaded PDF documents.
        * **Request:**
            * The request should be a `application/json` request.
            * The request body should contain a JSON object with a `query` field.
        * **Example using curl:**
            ```bash
            curl -X POST -H "Content-Type: application/json" -d '{"query": "What are the main findings?"}' [http://127.0.0.1:5000/search](https://www.google.com/search?q=http://127.0.0.1:5000/search)
            ```
        * **Response:**
            * A JSON response containing a list of relevant text snippets from the PDFs.
            * Example response:
                ```json
                {
                    "results": [
                        {"text": "Relevant text snippet 1...", "page": 1, "document": "document.pdf"},
                        {"text": "Relevant text snippet 2...", "page": 3, "document": "document.pdf"}
                    ]
                }
                ```

## How it Works

1.  **PDF Processing:** The `/upload` endpoint saves uploaded PDFs to the `pdfs/` directory, and the application extracts the text content.
2.  **Sentence Splitting:** The extracted text is split into sentences.
3.  **Embedding Generation:** Sentence Transformers are used to generate vector embeddings for each sentence.
4.  **FAISS Indexing:** The sentence embeddings are indexed using FAISS for efficient similarity search.
5.  **Search Query Embedding:** The `/search` endpoint converts the user's search query into a sentence embedding.
6.  **Similarity Search:** FAISS is used to find the most similar sentence embeddings to the query embedding.
7.  **Result Display:** The corresponding text snippets are returned in the JSON response.

## Notes
* This is a basic implementation and can be further improved with features like:
    * More advanced text processing.
    * More sophisticated ranking algorithms.
    * Better error handling.
    * Asynchronous processing.
