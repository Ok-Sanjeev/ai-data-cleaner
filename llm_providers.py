"""
Provider-Agnostic LLM Layer for AI Data Cleaner.

Abstracts LLM providers to support:
- Google Gemini (google-genai)
- OpenAI (and OpenAI-compatible endpoints: Groq, vLLM, Together AI, DeepSeek)
- Anthropic Claude (anthropic)
- Ollama (local, offline, air-gapped enterprise deployments)
- Mock Provider (for deterministic testing and CI/CD)
"""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


def extract_json_from_response(text: str) -> str:
    """Extract valid JSON from raw LLM output, stripping Markdown code fences if present."""
    text = text.strip()
    # Remove markdown code blocks if wrapped in ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    return text


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'openai', 'anthropic', 'ollama')."""
        pass

    @abstractmethod
    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        """Send prompt to LLM and return raw JSON string response."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider using google.genai."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(model_name=model_name, api_key=api_key)
        self._client = client
        if self._client is None:
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            from google import genai
            self._client = genai.Client(api_key=resolved_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def client(self) -> Any:
        return self._client

    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        from google.genai import types

        # Try structured schema generation first if schema_cls is provided
        if schema_cls is not None:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        response_mime_type="application/json",
                        response_schema=schema_cls,
                    ),
                )
                return extract_json_from_response(response.text or "")
            except Exception:
                pass  # Fall back to standard request

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )
        return extract_json_from_response(response.text or "")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI and OpenAI-compatible Provider.
    Works with OpenAI API or any OpenAI-compatible endpoint (Groq, vLLM, Together AI, LocalAI).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or "dummy-key"
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        super().__init__(model_name=model_name, api_key=resolved_key, base_url=resolved_base_url)

        from openai import OpenAI
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert data quality analyst and cleaning planner. "
                    "Analyze the dataset profile and return ONLY valid JSON matching the requested schema. "
                    "Do NOT include markdown formatting or extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return extract_json_from_response(content)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider."""

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
        super().__init__(model_name=model_name, api_key=resolved_key, base_url=base_url)

        from anthropic import Anthropic
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = Anthropic(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=(
                "You are an expert data quality analyst and cleaning planner. "
                "Analyze the dataset profile and return ONLY valid JSON matching the CleaningPlan schema. "
                "Do NOT wrap output in markdown fences or include commentary outside the JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        return extract_json_from_response(content)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Provider for air-gapped, offline enterprise environments.
    Connects to a local Ollama daemon (default: http://localhost:11434).
    Zero external internet access required.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        resolved_base_url = (
            base_url
            or os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        super().__init__(model_name=model_name, api_key=api_key, base_url=resolved_base_url)

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        import httpx

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert data quality analyst and cleaning planner. "
                        "Return ONLY valid JSON conforming to the CleaningPlan schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data.get("message", {}).get("content", "")
                return extract_json_from_response(content)
        except Exception as err:
            raise ConnectionError(
                f"Failed to communicate with local Ollama server at {url}. "
                f"Ensure Ollama is running (`ollama serve`). Error: {err}"
            ) from err


class MockProvider(BaseLLMProvider):
    """Deterministic Mock Provider for unit testing and offline development."""

    def __init__(
        self,
        model_name: str = "mock-planner",
        plan_json: Optional[str] = None,
        plan_dict: Optional[dict] = None,
    ):
        super().__init__(model_name=model_name)
        if plan_dict is not None:
            self._plan_json = json.dumps(plan_dict)
        elif plan_json is not None:
            self._plan_json = plan_json
        else:
            self._plan_json = json.dumps({
                "summary": "Mock cleaning plan",
                "actions": [],
            })
        self.call_history = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_plan(self, plan: Any) -> None:
        if isinstance(plan, dict):
            self._plan_json = json.dumps(plan)
        else:
            self._plan_json = str(plan)

    def generate_plan_json(self, prompt: str, schema_cls: Optional[Type] = None) -> str:
        self.call_history.append({"prompt": prompt, "schema": schema_cls})
        return self._plan_json


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Factory function to instantiate the configured LLM provider.
    Order of precedence:
      1. Explicit arguments
      2. Environment variables: LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL
      3. Auto-detection based on available API keys (Gemini -> OpenAI -> Anthropic -> Ollama)
    """
    resolved_provider = (
        provider_name
        or os.getenv("LLM_PROVIDER")
        or ("gemini" if os.getenv("GEMINI_API_KEY") else None)
        or ("openai" if os.getenv("OPENAI_API_KEY") else None)
        or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else None)
        or ("ollama" if os.getenv("OLLAMA_HOST") else "gemini")
    ).lower()

    resolved_model = model_name or os.getenv("LLM_MODEL")

    if resolved_provider == "gemini":
        model = resolved_model or "gemini-2.5-flash"
        return GeminiProvider(model_name=model, api_key=api_key)

    elif resolved_provider in {"openai", "vllm", "groq", "together"}:
        model = resolved_model or "gpt-4o-mini"
        return OpenAIProvider(model_name=model, api_key=api_key, base_url=base_url)

    elif resolved_provider == "anthropic":
        model = resolved_model or "claude-3-5-haiku-20241022"
        return AnthropicProvider(model_name=model, api_key=api_key, base_url=base_url)

    elif resolved_provider in {"ollama", "local"}:
        model = resolved_model or "llama3"
        return OllamaProvider(model_name=model, base_url=base_url, api_key=api_key)

    elif resolved_provider == "mock":
        model = resolved_model or "mock-model"
        return MockProvider(model_name=model)

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            f"Supported providers: 'gemini', 'openai', 'anthropic', 'ollama', 'mock'."
        )
