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
            return "[Error: chat_method not specified in registry]"

        # Resolve nested method path (e.g., chat.completions.create)
        method = self.client
        for attr in chat_method_name.split('.'):
            method = getattr(method, attr)

        call_kwargs = self.entry.get("extra_args", {}).copy()
        provider = self.entry["provider"].lower()
        
        # This is a good feature you added! Let's keep it.
        # It assumes you will add a "system_prompt" key to your registry entries.
        system_prompt_str = self.entry.get("system_prompt", "")

        # ---- Prepare API Arguments ----

        # Case 1: API uses a "messages" list (OpenAI, Anthropic, Mistral, XAI)
        if "messages" in call_kwargs:
            processed_messages = []
            for msg in call_kwargs["messages"]:
                content = msg.get("content", "")
                
                # Handle system prompt
                if msg.get("role") == "system":
                    if content == "{system_prompt}" and not system_prompt_str:
                        continue  # Skip empty system prompt
                    msg["content"] = content.replace("{system_prompt}", system_prompt_str)
                
                # Handle user prompt
                if "{prompt}" in content:
                    msg["content"] = content.replace("{prompt}", prompt)
                
                processed_messages.append(msg)
            call_kwargs["messages"] = processed_messages
        
        # Case 2: API uses a top-level "prompt" (Cohere, some HuggingFace)
        # Note: Google and HuggingFace positional args are handled below.
        elif provider not in {"google", "huggingface"}:
             call_kwargs["prompt"] = prompt

        # Add temperature for all providers except Google
        if provider != "google":
             call_kwargs["temperature"] = temp

        # ---- Call the API ----
        try:
            if provider == "google":
                # Google takes prompt positionally and temp via generation_config
                response = method(prompt)
            
            elif provider == "huggingface":
                # HuggingFace takes prompt positionally, others as kwargs
                response = method(prompt, **call_kwargs)
            
            else:
                # All other APIs (OpenAI, Anthropic, Mistral, Cohere, XAI)
                response = method(**call_kwargs)
        except Exception as e:
            message = f"API call failed for {self.model_alias}: {e}"
            self.logger.add_info(CHAT, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return f"[Error: API call failed for {self.model_alias}]"

        # ---- Extract text from response ----
        try:
            if provider == "google":
                return response.candidates[0].content.parts[0].text.strip()
            
            if provider == "anthropic":
                return response.content[0].text.strip()

            if provider == "openai" or provider == "mistral" or provider == "xai":
                return response.choices[0].message.content.strip()

            if provider == "cohere":
                return response.text.strip() # From a 'generate' call
            
            if provider == "huggingface":
                if isinstance(response, list) and "generated_text" in response[0]:
                    return response[0]["generated_text"].strip()
                return str(response).strip() # Fallback for text_generation
            
            # Fallbacks
            if hasattr(response, "text"):
                return response.text.strip()
            if isinstance(response, str):
                return response.strip()

            return str(response)
        except Exception as e:
            message = f"Failed to parse response from {self.model_alias}: {e}. Response: {str(response)[:200]}..."
            self.logger.add_info(CHAT, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return f"[Error: Could not parse response from {self.model_alias}]"

    def __call__(self, prompt: str, temperature: Optional[float] = None):
        return self.chat(prompt, temperature)
