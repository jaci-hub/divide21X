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
            self.logger.add_info(API, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return

        # Dynamic import
        module = importlib.import_module(registry_entry["import_module"])
        client_cls = getattr(module, registry_entry["client_class"])

        provider = registry_entry["provider"].lower()

        # ---- Special handling for Google Gemini ----
        if provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": self.temperature},
            )
        else:
            # Initialize client for all other providers
            init_kwargs = registry_entry.get("init_args", {})
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

        call_kwargs = self.entry.get("extra_args", {}).copy()

        # Replace {prompt} placeholders in messages
        if "messages" in call_kwargs:
            for msg in call_kwargs["messages"]:
                if "content" in msg and "{prompt}" in msg["content"]:
                    msg["content"] = msg["content"].replace("{prompt}", prompt)

        provider = self.entry["provider"].lower()

        # ---- 🔧 GOOGLE SPECIAL CASE ----
        if provider == "google":
            # Google’s API wants the prompt as a positional argument only
            response = method(prompt)
        else:
            call_kwargs["prompt"] = prompt
            call_kwargs["temperature"] = temp
            response = method(**call_kwargs)

        # ---- Extract text from response ----
        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "candidates"):
            try:
                return response.candidates[0].content.parts[0].text.strip()
            except Exception:
                return str(response)
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
