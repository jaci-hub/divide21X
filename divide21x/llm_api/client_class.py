import os
import json
import importlib
from typing import Optional

class ModelClient:
    def __init__(self, model_id: str, json_path: str = "divide21x/llm_api/registry.json"):
        """
        model_id: matches the "id" field in the JSON registry
        json_path: path to your JSON registry file
        """
        with open(json_path, "r") as f:
            self.registry = json.load(f)

        entry = next((m for m in self.registry if m["id"] == model_id), None)
        if entry is None:
            raise ValueError(f"Model with id '{model_id}' not found in registry.")

        self.entry = entry
        self.model_name = entry.get("model")
        self.temperature = entry.get("temperature", 0.0)
        self.api_key = os.environ.get(entry.get("api_key_env"))

        if not self.api_key:
            raise ValueError(f"API key for {entry['provider']} not found in environment variable {entry['api_key_env']}")

        # Dynamic import
        module = importlib.import_module(entry["import_module"])
        client_cls = getattr(module, entry["client_class"])

        # Initialize client
        init_kwargs = entry.get("init_args", {"api_key": self.api_key})
        self.client = client_cls(**init_kwargs)

    def chat(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Send a chat-like message using the dynamic chat method from JSON."""
        temp = temperature if temperature is not None else self.temperature

        # Prepare call args
        chat_method_name = self.entry.get("chat_method")
        if not chat_method_name:
            raise ValueError(f"chat_method not specified for {self.entry['id']}")

        method = getattr(self.client, chat_method_name)

        # Build kwargs dynamically
        call_kwargs = self.entry.get("extra_args", {}).copy()
        if system_prompt:
            call_kwargs["system_prompt"] = system_prompt
        call_kwargs["prompt"] = prompt
        call_kwargs["temperature"] = temp

        # Call the method
        response = method(**call_kwargs)

        # If the method returns an object with `.text` or `.content`, handle automatically
        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        elif isinstance(response, list) and len(response) > 0:
            return response[0]  # fallback for lists
        else:
            return str(response)

    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None):
        return self.chat(prompt, system_prompt, temperature)
