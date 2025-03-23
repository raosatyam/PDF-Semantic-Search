import os
import torch
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
import openai

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, OPENAI_API_KEY, GEMINI_API_KEY

class EmbeddingGenerator:
    def __init__(self, model_name=EMBEDDING_MODEL, embedding_dim=EMBEDDING_DIMENSION, use_openai=False, use_gemini=True):
        """
        Initialize the embedding generator.
        """

        self.model_name = model_name
        self.use_openai = use_openai and openai is not None and OPENAI_API_KEY
        self.use_gemini = self.model_name.startswith("gemini") and GEMINI_API_KEY is not None and use_gemini

        if self.use_gemini:
            genai.configure(api_key=GEMINI_API_KEY)
            self.embedding_dim = embedding_dim
        elif self.use_openai:
            openai.api_key = OPENAI_API_KEY
            self.embedding_dim = embedding_dim
        else:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings.
        """

        if not texts:
            return []

        if self.use_gemini:
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                return [np.array(embedding) for embedding in response["embedding"]]
            except Exception as e:
                print(f"Error generating Gemini embeddings: {str(e)}")
                self.use_gemini = False
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                return self.get_embeddings(texts)
        
        elif self.use_openai:
            try:
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [np.array(item.embedding) for item in response.data]
            except Exception as e:
                print(f"Error generating OpenAI embeddings: {str(e)}")
                self.use_openai = False
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                return self.get_embeddings(texts)
        else:
            # Use local embedding model
            embeddings = self.model.encode(texts)
            return [np.array(embedding) for embedding in embeddings]
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        """
        results = self.get_embeddings([text])
        return results[0] if results else np.zeros(self.embedding_dim)
    
    def generate_embedding_ids(self, count: int) -> List[str]:
        """
        Generate unique IDs for embeddings.
        """
        return [str(uuid.uuid4()) for _ in range(count)]