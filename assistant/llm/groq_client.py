"""
assistant/llm/groq_client.py — Resilient Groq client with retry, timeout,
and logging.

Model: llama-3.3-70b-versatile (best quality/speed tradeoff for this use
case). Temperature is kept low (0.25) deliberately — this is a financial
negotiation agent, not a creative-writing one; determinism matters more
than variety here.
"""
import os
import time
import logging
from typing import List, Dict, Optional

from groq import Groq, RateLimitError, APIConnectionError, APIStatusError

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_MAX_TOKENS = 1024
_TEMPERATURE = 0.25   # low = fewer hallucinations, more deterministic
_MAX_RETRIES = 3


class GroqLLMClient:
    """Thread-safe, resilient Groq LLM client."""

    def __init__(self) -> None:
        self.is_configured = bool(GROQ_API_KEY)
        if not self.is_configured:
            logger.warning(
                "GROQ_API_KEY not set — the agent will serve deterministic "
                "fallback replies instead of calling the LLM."
            )
        self._client = Groq(api_key=GROQ_API_KEY or "missing-key")

    def complete(
        self,
        messages: List[Dict],
        max_tokens: int = _MAX_TOKENS,
        temperature: float = _TEMPERATURE,
    ) -> str:
        """Calls the Groq API with exponential backoff on rate limits.
        Raises RuntimeError once retries are exhausted."""
        # Short-circuit when unconfigured: without a key every request is a
        # guaranteed 401, so skip the network round-trip entirely and let the
        # caller fall straight through to its deterministic path.
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY not configured — LLM disabled.")

        last_err: Exception = RuntimeError("No attempt made")

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content
                logger.debug("Groq OK - tokens used: %s", resp.usage.total_tokens if resp.usage else "?")
                return content.strip()

            except RateLimitError as exc:
                wait = 2 ** (attempt + 1)
                logger.warning("Groq rate-limited - waiting %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                last_err = exc

            except APIConnectionError as exc:
                logger.error("Groq connection lost: %s", exc)
                time.sleep(2)
                last_err = exc

            except APIStatusError as exc:
                logger.error("Groq API error %s: %s", exc.status_code, exc.message)
                last_err = exc
                break  # no retry on 4xx

            except Exception as exc:
                logger.exception("Unexpected Groq error")
                last_err = exc
                break

        raise RuntimeError(f"Groq unavailable after {_MAX_RETRIES} attempts: {last_err}")

    def complete_with_system(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Builds [system + history + user] and calls complete().
        `history` is a list of {"role": ..., "content": ...} dicts."""
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-20:])  # cap context window
        messages.append({"role": "user", "content": user_message})
        return self.complete(messages, **kwargs)


_instance: Optional[GroqLLMClient] = None


def get_groq_client() -> GroqLLMClient:
    global _instance
    if _instance is None:
        _instance = GroqLLMClient()
    return _instance
