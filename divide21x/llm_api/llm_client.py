import os
from typing import Optional, List, Dict, Any

#TODO

class ModelClient:
    def __init__(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(self._get_key_env())

        # if self.provider == "openai":
        #     from openai import OpenAI
        #     self.client = OpenAI(api_key=self.api_key)
        #     self.model = self.model or "gpt-4o"

        # elif self.provider == "anthropic":
        #     import anthropic
        #     self.client = anthropic.Anthropic(api_key=self.api_key)
        #     self.model = self.model or "claude-3-sonnet-20240229"

        # elif self.provider == "mistral":
        #     from mistralai.client import MistralClient
        #     self.client = MistralClient(api_key=self.api_key)
        #     self.model = self.model or "mistral-large-latest"

        # elif self.provider == "google":
        #     import google.generativeai as genai
        #     genai.configure(api_key=self.api_key)
        #     self.client = genai.GenerativeModel(self.model or "gemini-1.5-pro")

        # elif self.provider == "huggingface":
        #     from huggingface_hub import InferenceClient
        #     self.client = InferenceClient(token=self.api_key)
        #     self.model = self.model or "mistralai/Mixtral-8x7B-Instruct-v0.1"

        # else:
        #     raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_key_env(self) -> str:
        mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        return mapping.get(self.provider, "")

    def chat(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """Send a chat-like message to any supported LLM provider."""

        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=temperature,
                messages=messages,
            )
            return response.content[0].text.strip()

        elif self.provider == "mistral":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message["content"].strip()

        elif self.provider == "google":
            parts = [system_prompt, prompt] if system_prompt else [prompt]
            response = self.client.generate_content(parts)
            return response.text.strip()

        elif self.provider == "huggingface":
            response = self.client.text_generation(
                prompt,
                model=self.model,
                temperature=temperature,
                max_new_tokens=500,
            )
            return response.strip()

    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7):
        """Shortcut: model(prompt)"""
        return self.chat(prompt, system_prompt, temperature)
