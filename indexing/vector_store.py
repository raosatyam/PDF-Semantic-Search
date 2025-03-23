import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from config import FAISS_INDEX_FILE, METADATA_FILE, EMBEDDING_DIMENSION


class VectorStore:
    def __init__(self, index_file=FAISS_INDEX_FILE, metadata_file=METADATA_FILE, dimension=EMBEDDING_DIMENSION):
        """
        Initialize the vector store.
        """
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.dimension = dimension

        # Initialize or load index,Load metadata, Keep track of the next available ID
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
        self.next_id = len(self.metadata)

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        """
        Load existing FAISS index or create a new one.
        """
        if os.path.exists(self.index_file):
            try:
                return faiss.read_index(self.index_file)
            except Exception as e:
                print(f"Error loading FAISS index: {str(e)}. Creating new index.")

        # Create a new index
        return faiss.IndexFlatIP(self.dimension)

    def _load_metadata(self) -> Dict[int, Dict[str, Any]]:
        """
        Load metadata from file or initialize empty metadata.
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {str(e)}. Initializing empty metadata.")

        return {}

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        os.makedirs(os.path.dirname(self.metadata_file ), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _save_index(self) -> None:
        """Save FAISS index to file."""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)

    def add_embeddings(self, embeddings: List[np.ndarray], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """
        Add embeddings to the vector store.
        """
        if not embeddings:
            return []

        # Convert embeddings to float32 numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        # Get starting ID
        start_id = self.next_id

        # Add embeddings to FAISS index
        # Ensure embeddings are 2D and of correct shape
        if embeddings_array.shape[1] != self.index.d:
            raise ValueError(f"Embedding dimension mismatch: Expected {self.index.d}, but got {embeddings_array.shape[1]}")

        self.index.add(embeddings_array)

        # Create IDs for the new embeddings
        embedding_ids = [str(i) for i in range(start_id, start_id + len(embeddings))]

        # Store metadata
        for i, (embedding_id, metadata) in enumerate(zip(embedding_ids, metadata_list)):
            metadata["embedding_id"] = embedding_id
            self.metadata[embedding_id] = metadata

        # Update next ID
        self.next_id = start_id + len(embeddings)

        # Save metadata and index
        self._save_metadata()
        self._save_index()

        return embedding_ids

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        """
        if self.index.ntotal == 0:
            return []

        # Convert query to float32 numpy array and reshape
        query_embedding = np.array([query_embedding]).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Flatten results
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = []
        for i, idx in enumerate(indices):
            # Skip invalid indices (faiss returns -1 for empty results)
            if idx == -1:
                continue

            str_idx = str(idx)
            if str_idx in self.metadata:
                results.append({
                    "distance": distances[i],
                    "score": (1 + distances[i]) / 2,  
                    "metadata": self.metadata[str_idx]
                })

        return results

    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the vector store.
        """
        if embedding_id not in self.metadata:
            return False

        # Remove from metadata
        del self.metadata[embedding_id]
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from metadata."""
        # Create a new index
        new_index = faiss.IndexFlatIP(self.dimension)

        # No embeddings to add
        if not self.metadata:
            self.index = new_index
            self._save_index()
            return
        
        self._save_metadata()
        self._save_index()

    def get_metadata(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an embedding.
        """
        return self.metadata.get(embedding_id)