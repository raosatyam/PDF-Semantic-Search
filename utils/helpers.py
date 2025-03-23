import os
import re
import json
from typing import Dict, Any, List, Optional
from werkzeug.utils import secure_filename
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    """
    check if file has and allowed extenstion.
    """
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file) -> str:
    filename = secure_filename(file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path

def truncate_text_for_llm(text: str, max_tokens: int = 4000) -> str:
    """
    Truncate text to a maximum numbuer of tokens for LLM.
    """
    max_chars = max_tokens*4
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')

    if last_period > max_chars * 0.8:
        return truncated[:last_period+1]

    return truncated + "..."

def extract_snippets(text: str, query: str, context_size: int = 100) -> List[str]:
    """
    Extract snippets from text that contain the query terms.
    """
    # Convert query to regex pattern
    terms = re.findall(r'\b\w+\b', query.lower())
    pattern = '|'.join(re.escape(term) for term in terms)

    # Find all matches
    matches = list(re.finditer(pattern, text.lower()))

    if not matches:
        return []

    snippets = []
    for match in matches:
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)

        # Extract snippet
        snippet = text[start:end]

        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        snippets.append(snippet)

    # Remove duplicates and sort by length
    unique_snippets = list(set(snippets))
    return sorted(unique_snippets, key=len)