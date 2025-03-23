import hashlib
from typing import List, Dict, Any, Optional, Tuple

from indexing.embeddings import EmbeddingGenerator
from indexing.vector_store import VectorStore
from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD


class SemanticSearch:
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore,
                 top_k: int = TOP_K_RESULTS, threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize the semantic search engine.

        Args:
            embedding_generator: Embedding generator
            vector_store: Vector store
            top_k: Number of results to return
            threshold: Minimum similarity score to consider a match
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a semantic search for a query.
        """

        query_embedding = self.embedding_generator.get_embedding(query)
        results = self.vector_store.search(query_embedding, self.top_k)
        filtered_results = [result for result in results if result["score"] >= self.threshold]
        return filtered_results

    def get_query_hash(self, query: str) -> str:
        """
        Generate a hash for a query string.
        """
        return hashlib.md5(query.encode()).hexdigest()

    def determine_llm_need(self, results: List[Dict[str, Any]]) -> bool:
        """
        Determine if an LLM call is needed based on search results.
        """

        # TO DO: think about better logi

        if not results:
            return True

        # Check if the top result is highly relevant (very high score)
        if results and results[0]["score"] > 0.9:
            return False

        if len(results) > 1:
            top_score = results[0]["score"]
            second_score = results[1]["score"]

            if (top_score - second_score) < 0.05:
                return True

        return True