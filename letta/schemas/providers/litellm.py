"""LiteLLM proxy provider.

LiteLLM exposes an OpenAI-compatible /v1/chat/completions endpoint, but its
/v1/models response uses the minimal OpenAI shape (id/object/created/owned_by)
without context-window metadata. The richer metadata (max_model_len,
max_input_tokens, cost info, etc.) is on the LiteLLM-specific /model/info
endpoint at the proxy root (NOT under /v1).

This provider hits /model/info during discovery so context_window can be set
correctly per model, then routes inference through the standard OpenAI client
to /v1/chat/completions.
"""

from typing import Literal

import httpx
from pydantic import Field

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider


class LiteLLMProvider(Provider):
    provider_type: Literal[ProviderType.litellm] = Field(ProviderType.litellm, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.base, description="The category of the provider (base or byok)")
    base_url: str = Field(..., description="LiteLLM proxy root URL (e.g., http://localhost:4000). May or may not include trailing /v1.")
    api_key: str | None = Field(None, description="LiteLLM master key or virtual key (Bearer token).")
    handle_base: str | None = Field(None, description="Custom handle base name for model handles (e.g., 'gaia' instead of 'litellm').")

    def _root_url(self) -> str:
        """Return the proxy root URL without a trailing /v1 — /model/info lives here."""
        url = self.base_url.rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def _inference_url(self) -> str:
        """OpenAI-compatible inference URL with /v1 suffix."""
        return self._root_url() + "/v1"

    async def list_llm_models_async(self) -> list[LLMConfig]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self._root_url()}/model/info", headers=headers)
            resp.raise_for_status()
            entries = resp.json().get("data", [])

        configs = []
        for entry in entries:
            model_name = entry.get("model_name")
            if not model_name:
                continue
            info = entry.get("model_info") or {}
            # LiteLLM emits max_model_len (vLLM convention) and/or max_input_tokens
            # (OpenAI convention) on /model/info; fall back to a sane default.
            context_window = info.get("max_model_len") or info.get("max_input_tokens") or 32768

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint=self._inference_url(),
                    context_window=context_window,
                    handle=self.get_handle(model_name, base_name=self.handle_base) if self.handle_base else self.get_handle(model_name),
                    max_tokens=self.get_default_max_output_tokens(model_name),
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )
        return configs

    async def list_embedding_models_async(self) -> list[EmbeddingConfig]:
        # Embedding models could theoretically be exposed via the same proxy,
        # but discovery + cost-tracking semantics differ enough that we leave
        # them out for now; users can wire embeddings separately if needed.
        return []
