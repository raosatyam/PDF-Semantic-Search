from typing import List, Dict, Any, Optional
from llm.llm_manager import LLMManager

class TextSummarizer:
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize
        """
        self.llm_manager = llm_manager

    def summarize(self, text: str, detail_level: str = "mediyum") -> str:
        """
        Summarize text using an LLM.
        """

        if detail_level == "short":
            system_prompt = (
                "Create a concise, bullet-point summary of the following text. "
                "Focus only on the most important points. Keep it under 100 words."
            )
        elif detail_level == "medium":
            system_prompt = (
                "Create a clear summary of the following text. "
                "Include key points and maintain important technical details. "
                "Use 200-300 words."
                ""
            )
        else:
            system_prompt = (
                "Create a comprehensive summary of the following text. "
                "Include all important points, technical details, and context. "
                "Use a structured format with sections."
            )
        
        user_promt = f"Please summarize the following text: \n\n{text}"
        summary = self.llm_manager.generate_response(
            prompt=user_promt,
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        return summary
    
    def needs_summary(self, text: str, max_length: int = 1000) -> bool:
        """
        Determine if text needs summarization based on length.
        """
        #TO DO
        return len(text) > max_length