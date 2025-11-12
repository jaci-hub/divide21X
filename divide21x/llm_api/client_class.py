import os
import importlib
from typing import Optional
from divide21x.utils.logger import EpisodeLogger

# base dir
BASE_DIR = './divide21x/llm_api/logs'
# categories
MODEL = 'model'
API = 'api'
CHAT = 'chat'
# types
CRITICAL = 'critical'
WARNING = 'warning'


class ModelClient:
    def __init__(self, registry_entry=None):
        """
        model_id: matches the "id" field in the JSON registry
        json_path: path to your JSON registry file
        """
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
        if registry_entry is None:
            message = "No entry from registry.json provided."
            print(message)
            self.logger.add_info(MODEL, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

        self.entry = registry_entry
        self.model_name = registry_entry.get("model")
        self.model_alias = registry_entry.get("alias")
        self.temperature = registry_entry.get("temperature", 0.0)
        self.api_key = os.environ.get(registry_entry.get("api_key_env"))
        self.client = None

        if not self.api_key:
            message = (
                f"API key for {registry_entry['provider']} not found in "
                f"environment variable {registry_entry['api_key_env']}"
            )
            print(message)
            self.logger.add_info(API, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

        # Dynamic import
        module = importlib.import_module(registry_entry["import_module"])
        client_cls = getattr(module, registry_entry["client_class"])

        # Initialize client
        init_kwargs = registry_entry.get("init_args", {})
        # Handle Google’s GenerativeModel or others without api_key in init args
        if "api_key" in init_kwargs:
            init_kwargs["api_key"] = self.api_key
        elif registry_entry["provider"].lower() == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            init_kwargs["model_name"] = registry_entry["model"]
        else:
            init_kwargs["api_key"] = self.api_key

        self.client = client_cls(**init_kwargs)

    def chat(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send a chat-like message using the dynamic chat method from JSON."""
        temp = temperature if temperature is not None else self.temperature

        chat_method_name = self.entry.get("chat_method")
        if not chat_method_name:
            message = f"chat_method not specified for {self.entry['id']}"
            self.logger.add_info(CHAT, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

        # Support nested chat method paths like "chat.completions.create"
        method = self.client
        for attr in chat_method_name.split('.'):
            method = getattr(method, attr)

        # Build kwargs dynamically
        call_kwargs = self.entry.get("extra_args", {}).copy()

        # Replace any placeholder in extra_args with the actual prompt
        if "messages" in call_kwargs:
            for msg in call_kwargs["messages"]:
                if "content" in msg and "{prompt}" in msg["content"]:
                    msg["content"] = msg["content"].replace("{prompt}", prompt)

        # Add direct args for models that don't use messages
        call_kwargs["prompt"] = prompt
        call_kwargs["temperature"] = temp

        # Call the method
        response = method(**call_kwargs)

        # Return text content based on object type
        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        elif isinstance(response, list) and len(response) > 0:
            return str(response[0])
        else:
            return str(response)

    def __call__(self, prompt: str, temperature: Optional[float] = None):
        return self.chat(prompt, temperature)
