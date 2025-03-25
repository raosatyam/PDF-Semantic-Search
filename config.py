import os
from dotenv import load_dotenv
import shutil

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/text-embedding-004")
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

ALLOWED_EXTENSIONS = {'pdf'}


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
MAX_TOKENS = 500

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

INDEX_PATH = os.path.join(os.getcwd(), "index")
FAISS_INDEX_FILE = os.path.join(INDEX_PATH, "document_index.faiss")
METADATA_FILE = os.path.join(INDEX_PATH, "metadata.json")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

def reset_directory(path):
    if os.path.exists(path): 
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

reset_directory(UPLOAD_FOLDER)
reset_directory(INDEX_PATH)


CACHE_EXPIRATION = 36000
CACHE_ENABLED = True

DATABASE_NAME = os.getenv("DATABASE_NAME")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")