"""
assistant/llm/groq_client.py — Resilient Groq client with retry, timeout,
and logging.

Model: llama-3.3-70b-versatile (best quality/speed tradeoff for this use
case). Temperature is kept low (0.25) deliberately — this is a financial
negotiation agent, not a creative-writing one; determinism matters more
than variety here.

Latency is bounded: every call has a total time budget
(LLM_TOTAL_TIMEOUT_SECONDS) covering all attempts and backoff, and each HTTP
attempt its own timeout. The SDK's built-in retries are disabled so they
can't stack on top of ours. When the budget runs out, complete() raises and
callers fall back to deterministic replies — a slow LLM costs the client a
few seconds at most, never a hung request.
"""
import time
import logging
from typing import List, Dict, Optional

from groq import Groq, RateLimitError, APIConnectionError, APIStatusError

import metrics
from config import GROQ_API_KEY, GROQ_MODEL, LLM_REQUEST_TIMEOUT_SECONDS, LLM_TOTAL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_MAX_TOKENS = 1024
_TEMPERATURE = 0.25   # low = fewer hallucinations, more deterministic
_MAX_ATTEMPTS = 3
_MIN_USEFUL_ATTEMPT_SECONDS = 1.0  # don't start an attempt with less time than this left


class GroqLLMClient:
    """Thread-safe, resilient Groq LLM client."""

    def __init__(self) -> None:
        self.is_configured = bool(GROQ_API_KEY)
        if not self.is_configured:
            logger.warning(
                "GROQ_API_KEY not set — the agent will serve deterministic "
                "fallback replies instead of calling the LLM."
            )
        self._client = Groq(
            api_key=GROQ_API_KEY or "missing-key",
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,  # retries are handled below, within the time budget
        )

    def complete(
        self,
        messages: List[Dict],
        max_tokens: int = _MAX_TOKENS,
        temperature: float = _TEMPERATURE,
        budget_seconds: Optional[float] = None,
    ) -> str:
        """Calls the Groq API, retrying transient failures with backoff as
        long as the time budget allows. Raises RuntimeError on failure."""
        # Short-circuit when unconfigured: without a key every request is a
        # guaranteed 401, so skip the network round-trip entirely and let the
        # caller fall straight through to its deterministic path.
        if not self.is_configured:
            metrics.LLM_REQUESTS.labels(outcome="not_configured").inc()
            raise RuntimeError("GROQ_API_KEY not configured — LLM disabled.")

        budget = budget_seconds if budget_seconds is not None else LLM_TOTAL_TIMEOUT_SECONDS
        start = time.monotonic()
        deadline = start + budget
        last_err: Exception = RuntimeError("No attempt made within the time budget")

        try:
            for attempt in range(_MAX_ATTEMPTS):
                remaining = deadline - time.monotonic()
                if remaining < _MIN_USEFUL_ATTEMPT_SECONDS:
                    break
                try:
                    resp = self._client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=min(LLM_REQUEST_TIMEOUT_SECONDS, remaining),
                    )
                    content = resp.choices[0].message.content
                    logger.debug("Groq OK - tokens used: %s", resp.usage.total_tokens if resp.usage else "?")
                    metrics.LLM_REQUESTS.labels(outcome="ok").inc()
                    return content.strip()

                except RateLimitError as exc:
                    last_err = exc
                    backoff = 2 ** attempt
                    logger.warning("Groq rate-limited (attempt %d)", attempt + 1)

                except APIConnectionError as exc:  # includes timeouts
                    last_err = exc
                    backoff = 1
                    logger.warning("Groq connection error or timeout (attempt %d): %s", attempt + 1, type(exc).__name__)

                except APIStatusError as exc:
                    logger.error("Groq API error %s", exc.status_code)
                    last_err = exc
                    break  # no retry on other 4xx/5xx

                except Exception as exc:
                    logger.exception("Unexpected Groq error")
                    last_err = exc
                    break

                if attempt == _MAX_ATTEMPTS - 1:
                    break
                if time.monotonic() + backoff + _MIN_USEFUL_ATTEMPT_SECONDS >= deadline:
                    break  # no time left for another useful attempt
                time.sleep(backoff)
        finally:
            metrics.LLM_LATENCY.observe(time.monotonic() - start)

        metrics.LLM_REQUESTS.labels(outcome="error").inc()
        raise RuntimeError(f"Groq unavailable within {budget:.0f}s: {type(last_err).__name__}")

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
