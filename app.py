import traceback
import threading
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import json
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from indexing.document_parser import DocumentParser
from indexing.embeddings import EmbeddingGenerator
from indexing.vector_store import VectorStore
from search.semantic_search import SemanticSearch
from llm.llm_manager import LLMManager
from utils.helpers import allowed_file, save_uploaded_file
from utils.cache import ResponseCache
from search.query_processor import QueryProcessor
from database.db_manager import DatabaseManager

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB max upload size

# Initialize components
document_parser = DocumentParser()
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
llm_manager = LLMManager()
db_manager = DatabaseManager()
cache = ResponseCache(db_manager)
search_engine = SemanticSearch(embedding_generator, vector_store)
query_processor = QueryProcessor(search_engine, llm_manager, cache)

# In-memory document storage
documents = {}
document_id_counter = 1


@app.route('/')
def index():
    return jsonify({
        "msg" : "Welcome!!!"
    })
    

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a PDF file."""
    global document_id_counter

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    if file:
        try:
            file_path = save_uploaded_file(file)
            document_data = document_parser.process_document(file_path)

            # Generate document ID
            document_id = document_id_counter
            document_id_counter += 1

            documents[document_id] = {
                'id': document_id,
                'filename': document_data['filename'],
                'path': file_path,
                'title': document_data['title'],
                'page_count': document_data['page_count'],
                'chunks': [],
                'indexed': False
            }

            all_chunks = document_data['chunks']
            chunk_texts = [chunk['content'] for chunk in all_chunks]
            chunk_embeddings = embedding_generator.get_embeddings(chunk_texts)

            # Prepare metadata for each chunk
            metadata_list = []
            for i, chunk in enumerate(all_chunks):
                metadata = {
                    'document_id': document_id,
                    'document_title': document_data['title'],
                    'content': chunk['content'],
                    'page_number': chunk['page_number'],
                    'chunk_index': chunk['chunk_index']
                }
                metadata_list.append(metadata)

            # Add embeddings to vector store
            embedding_ids = vector_store.add_embeddings(chunk_embeddings, metadata_list)

            # Store chunks in memory instead of database
            for i, chunk in enumerate(all_chunks):
                chunk_info = {
                    'chunk_index': chunk['chunk_index'],
                    'page_number': chunk['page_number'],
                    'content': chunk['content'],
                    'embedding_id': embedding_ids[i]
                }
                documents[document_id]['chunks'].append(chunk_info)

            # Mark document as indexed
            documents[document_id]['indexed'] = True

            # To clear all the cache present in the db
            db_manager.clean_all_cache()

            return jsonify({
                'success': True,
                'message': f'File {file.filename} uploaded and indexed successfully',
                'document_id': document_id
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/search', methods=['POST'])
def search():
    """Search for documents."""
    data = request.json
    query = data.get('query', '').strip()
    
    detail_level = data.get('detail_level', 'medium')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        result = cache.get_cached_response(query, data)
        if result:
            result["response_type"] = "Cache"
            return jsonify(result)
    except:
        traceback.print_exc()
        print("error in retriving from cache")
        pass

    try:
        # Process the query
        result = query_processor.process_query(query, detail_level)
        cache_thread = threading.Thread(target=cache.cache_response, args=(query, result, data), daemon=True)
        cache_thread.start()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents."""
    try:
        document_list = [doc for doc_id, doc in documents.items()]
        return jsonify({'documents': document_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/documents/<int:document_id>', methods=['GET'])
def get_document(document_id):
    """Get a specific document."""
    try:
        document = documents.get(document_id)
        if document:
            return jsonify({'document': document})
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics."""
    try:
        return jsonify({
            'documents_count': len(documents),
            'cache_hits': 0  # Removed cache tracking
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5007)