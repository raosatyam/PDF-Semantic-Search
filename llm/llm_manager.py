import os
from typing import Dict, Any, List, Optional
import openai
import google.generativeai as genai
from time import sleep

from config import LLM_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, LLM_MODEL, MAX_TOKENS 

class LLMManager:
    def __init__(self, provider=LLM_PROVIDER, model=LLM_MODEL, max_tokens=MAX_TOKENS):
        """
        Initialze the LLM Manager
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

        if provider == "google":
            genai.configure(api_key=GEMINI_API_KEY)
        elif provider == "openai":
            openai.api_key = OPENAI_API_KEY
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_response(self, prompt: str, temperature: float = 0.7,
                          system_prompt: Optional[str] = None) -> str:
        """
        Generate a respomse using the LLM 
        """

        if self.provider == "openai":
            return self._generate_openai_response(prompt, temperature, system_prompt)
        elif self.provider == "google":
            return self._generate_gemini_response(prompt, temperature, system_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_gemini_response(self, prompt: str, temperature: float,
                                  system_prompt: Optional[str] = None) -> str:
        """
        Generate response using Gemini API
        """

        messages = []
        if system_prompt:
            messages.append(system_prompt)
        messages.append(prompt)

        max_retries = 2
        for attemt in range(1, max_retries+1):
            try:
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    messages,
                    generation_config={"temperature": temperature, "max_output_tokens": self.max_tokens}
                )
                generated_text = response.text.strip() if response else ""
                return generated_text
            except Exception as e:
                if attemt < max_retries:
                    print(f" Gemini API error: {str(e)}. Retrying in {attemt**2} seconds...")
                    sleep(attemt**2)
                else:
                    raise

    def _generate_openai_response(self, prompt: str, temperature: float,
                                  system_prompt: Optional[str]) -> str:
        """
        Generate response using OPENAI API
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 2
        for attemt in range(1, max_retries+1):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attemt < max_retries:
                    print(f" Gemini API error: {str(e)}. Retrying in {attemt**2} seconds...")
                    sleep(attemt**2)
                else:
                    raise