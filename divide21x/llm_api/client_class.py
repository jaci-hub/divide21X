import os
import json
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
            message = f"No entry from registry.json provided."
            self.logger.add_info(MODEL, CRITICAL, message)
            # log
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
            message = f"API key for {registry_entry['provider']} not found in environment variable {registry_entry['api_key_env']}"
            self.logger.add_info(API, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

        # Dynamic import
        module = importlib.import_module(registry_entry["import_module"])
        client_cls = getattr(module, registry_entry["client_class"])

        # Initialize client
        init_kwargs = registry_entry.get("init_args", {"api_key": self.api_key})
        self.client = client_cls(**init_kwargs)

    def chat(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Send a chat-like message using the dynamic chat method from JSON."""
        temp = temperature if temperature is not None else self.temperature

        # Prepare call args
        chat_method_name = self.entry.get("chat_method")
        if not chat_method_name:
            message = f"chat_method not specified for {self.entry['id']}"
            self.logger.add_info(CHAT, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

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
