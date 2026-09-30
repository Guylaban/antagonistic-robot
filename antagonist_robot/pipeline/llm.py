"""Provider-agnostic LLM client using the openai Python SDK.

Works with any OpenAI-compatible API (Grok, OpenAI, Groq, Together, Ollama)
by changing base_url and model in config. The same code handles all providers.
"""

import time
from typing import Dict, List

from openai import OpenAI

from antagonist_robot.config.settings import LLMConfig
from antagonist_robot.pipeline.types import LLMResult


class LLMEngine:
    """LLM text generation via any OpenAI-compatible API.

    Initialized once with config. The generate method sends a system prompt
    and conversation history, and returns the complete response.
    """

    def __init__(self, config: LLMConfig):
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> LLMResult:
        """Send messages to LLM and return the full response.

        Uses non-streaming mode. The complete response is returned
        after the LLM finishes generating.

        Args:
            system_prompt: The system message (from hostility manager).
            messages: Conversation history as list of
                      {"role": "user"|"assistant", "content": str}.

        Returns:
            LLMResult with response text, model name, token count, and timing.
        """
        start = time.monotonic()

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=False,
        )

        elapsed = time.monotonic() - start
        choice = response.choices[0]
        usage = response.usage

        # Some providers (e.g. xAI reasoning models) return a separate
        # reasoning trace; keep it so it can be stored apart from the response.
        extra = getattr(choice.message, "model_extra", None) or {}
        reasoning = getattr(choice.message, "reasoning_content", None) or extra.get("reasoning_content")

        return LLMResult(
            text=(choice.message.content or "").strip(),
            model=response.model,
            total_tokens=usage.total_tokens if usage else 0,
            generation_time_seconds=elapsed,
            reasoning=reasoning or None,
        )
