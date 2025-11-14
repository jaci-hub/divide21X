import os
import importlib
import traceback
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
        Dynamic client wrapper that initializes per the registry entry.
        This handles provider-specific constructors and import name quirks.
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

        provider = registry_entry["provider"].lower()

        # Try provider-aware import and initialization
        try:
            # Special cases first
            if provider == "google":
                # Google handled with generativeai library
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": self.temperature},
                )
                return

            if provider == "openai":
                # Modern OpenAI client auto-loads key from env var
                # Ensure env var set and instantiate OpenAI()
                try:
                    # set env var for OpenAI client to read
                    os.environ["OPENAI_API_KEY"] = self.api_key
                    from openai import OpenAI
                    self.client = OpenAI()
                    return
                except Exception:
                    # fallback to dynamic import path if different version
                    pass

            # For xAI (Grok), PyPI package xai-sdk exposes package `xai`
            if provider == "xai":
                # try common import names
                for mod_name in ("xai", "xai_sdk"):
                    try:
                        module = importlib.import_module(mod_name)
                        client_cls = getattr(module, registry_entry.get("client_class", "Client"))
                        # many versions use Client(api_key=...) or Client(...)
                        try:
                            self.client = client_cls(api_key=self.api_key)
                        except TypeError:
                            # maybe takes token or key as first positional arg
                            self.client = client_cls(self.api_key)
                        return
                    except Exception:
                        continue
                raise ImportError("xAI SDK import failed for both 'xai' and 'xai_sdk'")

            # Generic dynamic import for other providers
            import_module_name = registry_entry.get("import_module")
            # some registry entries might list a PyPI name instead of import; try sensible fallbacks
            tried = []
            module = None
            if import_module_name:
                for candidate in [import_module_name, import_module_name.replace('-', '_'), import_module_name.replace('-', '')]:
                    tried.append(candidate)
                    try:
                        module = importlib.import_module(candidate)
                        break
                    except Exception:
                        module = None
                if module is None:
                    raise ModuleNotFoundError(f"Could not import any of {tried}")

            client_cls_name = registry_entry.get("client_class")
            if not module or not client_cls_name:
                raise ImportError("Missing module or client class in registry entry")

            client_cls = getattr(module, client_cls_name)

            # Build init kwargs with provider-specific parameter names
            init_kwargs = registry_entry.get("init_args", {}).copy() or {}

            # Provider-specific arg name mapping
            if provider in {"huggingface", "huggingface_hub"}:
                # InferenceClient expects token=...
                init_kwargs.setdefault("token", self.api_key)
            elif provider == "cohere":
                # cohere.Client accepts api_key positional or api_key kw
                init_kwargs.setdefault("api_key", self.api_key)
            elif provider == "anthropic":
                init_kwargs.setdefault("api_key", self.api_key)
            elif provider == "mistral":
                init_kwargs.setdefault("api_key", self.api_key)
            else:
                # default try api_key
                init_kwargs.setdefault("api_key", self.api_key)

            # Instantiate client
            try:
                self.client = client_cls(**init_kwargs)
            except TypeError as e:
                # Try positional fallback (some clients want token positional)
                try:
                    self.client = client_cls(self.api_key)
                except Exception as e2:
                    raise e  # re-raise original to be caught below

        except Exception as exc:
            # Log full traceback for debugging in CI logs
            tb = traceback.format_exc()
            message = f"Failed to initialize client for {self.model_alias} ({provider}): {exc}\n{tb}"
            # print also to stdout so it shows in GitHub Actions logs immediately
            print(message)
            self.logger.add_info(MODEL, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            self.client = None
            return

    def chat(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send a chat-like message using the dynamic chat method from JSON."""
        temp = temperature if temperature is not None else self.temperature

        if self.client is None:
            message = f"No client initialized for {self.model_alias}"
            self.logger.add_info(CHAT, CRITICAL, message)
            return f"[Error: no client for {self.model_alias}]"

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

        system_prompt_str = self.entry.get("system_prompt", "")

        # Prepare messages/prompt
        if "messages" in call_kwargs:
            processed_messages = []
            for msg in call_kwargs["messages"]:
                content = msg.get("content", "")
                if msg.get("role") == "system":
                    if content == "{system_prompt}" and not system_prompt_str:
                        continue
                    msg["content"] = content.replace("{system_prompt}", system_prompt_str)
                if "{prompt}" in content:
                    msg["content"] = content.replace("{prompt}", prompt)
                processed_messages.append(msg)
            call_kwargs["messages"] = processed_messages
        elif provider not in {"google", "huggingface", "huggingface_hub", "cohere"}:
            call_kwargs["prompt"] = prompt

        if provider != "google":
            call_kwargs["temperature"] = temp

        # Call the API and capture errors with tracebacks for CI logs
        try:
            if provider == "google":
                response = method(prompt)
            elif provider in {"huggingface", "huggingface_hub"}:
                response = method(prompt, **call_kwargs)
            else:
                response = method(**call_kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            message = f"API call failed for {self.model_alias} ({provider}): {e}\n{tb}"
            print(message)
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
                content = getattr(response, "content", None)
                if isinstance(content, list) and len(content) > 0 and hasattr(content[0], "text"):
                    return content[0].text.strip()

            # OpenAI, Mistral, and xAI all use the same OpenAI-compatible response structure
            if provider in {"openai", "mistral", "xai"}:
                if hasattr(response, "choices") and len(response.choices) > 0:
                    if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
                        return response.choices[0].message.content.strip()

            if provider == "cohere":
                if hasattr(response, "text"):
                    return response.text.strip()

            if provider in {"huggingface", "huggingface_hub"}:
                if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                    if "generated_text" in response[0]:
                        return response[0]["generated_text"].strip()

            # Generic fallbacks
            if hasattr(response, "text"):
                return response.text.strip()
            if isinstance(response, str):
                return response.strip()
            return str(response)
        except Exception as e:
            tb = traceback.format_exc()
            message = f"Failed to parse response from {self.model_alias}: {e}\n{tb}\nResponse repr: {repr(response)[:400]}"
            print(message)
            self.logger.add_info(CHAT, CRITICAL, message)
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            self.logger.save_episode()
            return f"[Error: Could not parse response from {self.model_alias}]"

    def __call__(self, prompt: str, temperature: Optional[float] = None):
        return self.chat(prompt, temperature)
