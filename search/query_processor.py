import json
from typing import List, Dict, Any, Optional, Tuple
import re

from search.semantic_search import SemanticSearch
from llm.llm_manager import LLMManager
from llm.summarization import TextSummarizer
from llm.rephrasing import TextRephraser
from utils.helpers import truncate_text_for_llm, extract_snippets
from utils.cache import ResponseCache


class QueryProcessor:
    def __init__(self, search_engine: SemanticSearch, llm_manager: LLMManager, cache: ResponseCache):
        """
        Initialize the query processor.
        """
        self.search_engine = search_engine
        self.llm_manager = llm_manager
        self.summarizer = TextSummarizer(llm_manager)
        self.rephraser = TextRephraser(llm_manager)
        self.cache = cache

    def process_query(self, query: str, detail_level: str = "medium") -> Dict[str, Any]:
        """
        Process a query and return the result.
        """

        cache_key = f"{query}_{detail_level}"
        cached_result = self.cache.get_cached_response(cache_key)
        if cached_result:
            return json.loads(cached_result)

        search_results = self.search_engine.search(query)
        need_llm = self.search_engine.determine_llm_need(search_results)

        result = {
            "query": query,
            "results": search_results,
            "used_llm": need_llm,
            "detail_level": detail_level
        }

        # No results, generate a fallback response
        if not search_results:
            result["response"] = self._generate_fallback_response(query)
            result["used_llm"] = True
            result["response_type"] = "fallback"

        # Direct response from search results
        elif not need_llm:
            best_result = search_results[0]
            content = best_result["metadata"].get("content", "")

            # Check if content needs summarization
            if self.summarizer.needs_summary(content) and detail_level != "detailed":
                content = self.summarizer.summarize(content, detail_level)
                result["response_type"] = "summarized"
            else:
                if content[0].islower():
                    cutoff_index = next((i for i, char in enumerate(content) if char in ".?!"), None)
                    if cutoff_index is not None and cutoff_index < len(content) // 4:
                        content = content[cutoff_index+1:].strip()

                result["response_type"] = "direct"

            result["response"] = content

        # Enhanced response with LLM
        else:
            # Combine relevant passages
            combined_text = self._combine_relevant_passages(search_results, query)

            # Generate enhanced response
            response = self._generate_enhanced_response(combined_text, query, detail_level)

            result["response"] = response
            result["response_type"] = "enhanced"
        
        return result

    def _combine_relevant_passages(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Combine relevant passages from search results.
        """
        if not results:
            return ""

        passages = []
        for i, result in enumerate(results[:3]):
            content = result["metadata"].get("content", "")
            source = result["metadata"].get("document_title", "Unknown")
            page = result["metadata"].get("page_number", "")

            snippets = extract_snippets(content, query)

            if snippets:
                for snippet in snippets:
                    passages.append(f"[Source: {source}, Page: {page}] {snippet}")
            else:
                passages.append(f"[Source: {source}, Page: {page}] {content}")

        combined_text = "\n\n".join(passages)
        return truncate_text_for_llm(combined_text)

    def _generate_enhanced_response(self, text: str, query: str, detail_level: str) -> str:
        """
        Generate an enhanced response using LLM.
        """
        # Determine if we need to summarize, rephrase, or both
        needs_summary = self.summarizer.needs_summary(text)
        needs_rephrasing = self.rephraser.needs_rephrasing(text)

        system_prompt = (
            f"You are a helpful assistant that provides accurate answers based on the given context. "
            f"Answer the query using only the information provided in the context. "
            f"If the context doesn't contain relevant information, admit that you don't know. "
            f"Format your answer according to the detail level: {detail_level} "
            f"(short: concise bullet points, medium: balanced answer, detailed: comprehensive explanation). "
            f"Keep your answer factual and avoid adding information not present in the context."
        )

        user_prompt = f"Query: {query}\n\nContext:\n{text}"

        # Generate response
        response = self.llm_manager.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        return response

    def _generate_fallback_response(self, query: str) -> str:
        """
        Generate a fallback response when no results are found.
        """
        system_prompt = (
            "You are a helpful assistant. The user has asked a question about a document, "
            "but we couldn't find any relevant information in our database. "
            "Politely explain that you don't have specific information about their query, "
            "but offer general advice or suggestions about their topic if possible."
        )

        user_prompt = f"I need information about: {query}"

        # Generate fallback response
        response = self.llm_manager.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        return response