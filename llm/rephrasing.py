from typing import List, Dict, Any, Optional
from llm.llm_manager import LLMManager

import re
import nltk
from nltk.tokenize import sent_tokenize
from textstat import flesch_kincaid_grade

# Ensure required NLTK resources are available
nltk.download("punkt_tab")

class TextRephraser:
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the text rephraser.
        """
        self.llm_manager = llm_manager
        
        self.TECHNICAL_TERMS = {
            "hereby", "aforementioned", "hereinafter", "pursuant",
            "wherein", "therein", "thereto", "whereby", "whereas",
            "notwithstanding", "henceforth", "thereupon", "heretofore",
            "inasmuch", "aforestated"
        }

        self.PASSIVE_VOICE_PATTERN = r'\b(is|was|were|are|been|being) (\w+ed)\b'

        self.HIGH_COMPLEXITY_THRESHOLD = 30  # Average sentence length threshold
        self.READABILITY_THRESHOLD = 10  # Flesch-Kincaid Grade Level threshold
    
    def rephrase(self, text: str, query: str = None) -> str:
        """
        Rephrase text to make it clearer or more relevant to the query.
        """

        system_prompt = (
            "Rephrase the following text to make it clearer and more readable. "
            "Maintain all important technical details and information. "
            "Do not add new information not present in the original text."
        )

        if query:
            user_prompt = (
                f"Please rephrase the following text to make it more relevant "
                f"to the question: '{query}'\n\n{text}"
            )
        else:
            user_prompt = f"Please rephrase the following text to make it clearer:\n\n{text}"

        rephrased = self.llm_manager.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more deterministic output
        )
        return rephrased
    
    def needs_rephrasing(self, text: str) -> bool:
        """
        Determine if text needs rephrasing based on readability, complexity, and style.

        Args:
            text: Text to check.

        Returns:
            True if rephrasing is recommended, False otherwise.
        """
        # Tokenize sentences properly
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Compute readability score (Flesch-Kincaid Grade Level)
        readability_score = flesch_kincaid_grade(text)

        # Check for technical jargon
        has_technical_jargon = any(term in text.lower() for term in self.TECHNICAL_TERMS)

        # Detect passive voice using regex
        has_passive_voice = bool(re.search(self.PASSIVE_VOICE_PATTERN, text.lower()))

        # Determine if text needs rephrasing
        return (avg_sentence_length > self.HIGH_COMPLEXITY_THRESHOLD or
                readability_score > self.READABILITY_THRESHOLD or
                has_technical_jargon or
                has_passive_voice)
