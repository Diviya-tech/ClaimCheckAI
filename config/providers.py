"""LLM provider abstraction layer with soft cost guardrails.

Every LLM call in ClaimCheck AI goes through `llm_call`. Pipeline code never
imports `anthropic` (or `openai`) directly — it asks for a model *tier*
("premium" or "lightweight") and this module routes the request to the
configured provider. To add OpenAI later, implement another `LLMProvider`
subclass and register it in `_PROVIDERS`; no pipeline code changes.

Responsibilities:
  * Route a tier -> concrete provider + model (from config.settings.MODEL_CONFIG).
  * Optionally return structured output via tool_use when `response_schema` is given.
  * Track daily token usage in a JSON file and enforce a *soft* daily budget.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger("claimcheck.providers")


# --------------------------------------------------------------------------- #
# Soft budget guardrail
# --------------------------------------------------------------------------- #
class BudgetWarning(Exception):
    """Raised when the daily token budget is already exhausted.

    This is a *soft* stop: callers may catch it and re-invoke `llm_call` with
    `allow_over_budget=True` to proceed anyway. It is an Exception (not a
    warnings.Warning) so it interrupts control flow and forces a deliberate
    decision rather than being silently swallowed.
    """


# --------------------------------------------------------------------------- #
# Daily usage tracking (persisted to a small JSON file)
# --------------------------------------------------------------------------- #
def _usage_path() -> Path:
    return Path(settings.DAILY_USAGE_FILE)


def _load_usage() -> dict[str, dict[str, int]]:
    """Load the usage ledger: {"YYYY-MM-DD": {input, output, total}, ...}."""
    path = _usage_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable ledger should never crash a pipeline run.
        logger.warning("Could not read %s; starting a fresh usage ledger.", path)
        return {}


def _save_usage(ledger: dict[str, dict[str, int]]) -> None:
    try:
        _usage_path().write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist token usage to %s: %s", _usage_path(), exc)


def get_daily_usage(day: str | None = None) -> int:
    """Return total tokens (input + output) used on `day` (defaults to today)."""
    day = day or date.today().isoformat()
    return _load_usage().get(day, {}).get("total", 0)


def _record_usage(input_tokens: int, output_tokens: int) -> int:
    """Add this call's tokens to today's tally and return the new daily total."""
    ledger = _load_usage()
    today = date.today().isoformat()
    entry = ledger.setdefault(today, {"input": 0, "output": 0, "total": 0})
    entry["input"] += input_tokens
    entry["output"] += output_tokens
    entry["total"] = entry["input"] + entry["output"]
    _save_usage(ledger)
    return entry["total"]


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Minimal provider contract. Implement once per backend (Anthropic, OpenAI...)."""

    @abstractmethod
    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        response_schema: dict[str, Any] | None,
        images: list[dict[str, str]] | None = None,
    ) -> tuple[str | dict[str, Any], int, int]:
        """Run one completion.

        Args:
            images: Optional provider-neutral image inputs, each a dict with
                ``media_type`` (e.g. "image/png") and base64 ``data``. The
                provider translates these into its own vision format. None for
                text-only calls.

        Returns (content, input_tokens, output_tokens) where `content` is a
        plain string, or — when `response_schema` is provided — the validated
        structured dict produced via tool_use.
        """
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude backend. Structured output is done via forced tool_use."""

    # Name of the synthetic tool used to coerce structured JSON output.
    _STRUCTURED_TOOL = "emit_structured_output"

    def __init__(self) -> None:
        # Imported lazily so importing this module doesn't require the SDK to be
        # installed until an Anthropic call is actually made.
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=settings.require_anthropic_key())

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        response_schema: dict[str, Any] | None,
        images: list[dict[str, str]] | None = None,
    ) -> tuple[str | dict[str, Any], int, int]:
        # Build the user message content. With images, content becomes a list of
        # image blocks followed by the text block; without, it's a plain string.
        if images:
            user_content: Any = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                }
                for img in images
            ]
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content = prompt

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if response_schema is not None:
            # Force the model to call a single tool whose input schema is the
            # caller's schema. The tool's `input` is then guaranteed to match.
            kwargs["tools"] = [
                {
                    "name": self._STRUCTURED_TOOL,
                    "description": "Return the result strictly matching the provided schema.",
                    "input_schema": response_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": self._STRUCTURED_TOOL}

        response = self._client.messages.create(**kwargs)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        if response_schema is not None:
            content = self._extract_tool_input(response)
        else:
            content = self._extract_text(response)

        return content, input_tokens, output_tokens

    def _extract_text(self, response: Any) -> str:
        parts = [block.text for block in response.content if block.type == "text"]
        return "".join(parts).strip()

    def _extract_tool_input(self, response: Any) -> dict[str, Any]:
        for block in response.content:
            if block.type == "tool_use" and block.name == self._STRUCTURED_TOOL:
                return block.input
        raise RuntimeError(
            "Expected a tool_use block for structured output but none was returned."
        )


# Registry of available providers. Add "openai": OpenAIProvider here later.
_PROVIDER_FACTORIES: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
}

# Providers are stateless-ish (hold a client); cache one instance per name.
_PROVIDER_CACHE: dict[str, LLMProvider] = {}


def _get_provider(name: str) -> LLMProvider:
    if name not in _PROVIDER_FACTORIES:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {sorted(_PROVIDER_FACTORIES)}"
        )
    if name not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[name] = _PROVIDER_FACTORIES[name]()
    return _PROVIDER_CACHE[name]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    model_tier: str = "premium",
    response_schema: dict[str, Any] | None = None,
    images: list[dict[str, str]] | None = None,
    allow_over_budget: bool = False,
) -> str | dict[str, Any]:
    """Run a single LLM completion through the configured provider.

    Args:
        prompt: The user message.
        system_prompt: Optional system instructions.
        model_tier: "premium" (Sonnet, reasoning) or "lightweight" (Haiku, cheap).
        response_schema: If given, a JSON Schema; the result is returned as a
            validated dict via tool_use instead of free text.
        images: Optional provider-neutral image inputs for vision calls, each a
            dict with ``media_type`` and base64 ``data``.
        allow_over_budget: If True, skip the soft budget check and proceed even
            when the daily budget is already exhausted.

    Returns:
        The completion text (str), or a structured dict when `response_schema`
        is provided.

    Raises:
        BudgetWarning: if today's usage already meets/exceeds DAILY_TOKEN_BUDGET
            and `allow_over_budget` is False. Catch and retry with
            allow_over_budget=True to continue.
        ValueError: if `model_tier` or the configured provider is unknown.
    """
    if model_tier not in settings.MODEL_CONFIG:
        raise ValueError(
            f"Unknown model_tier {model_tier!r}. "
            f"Available: {sorted(settings.MODEL_CONFIG)}"
        )

    # --- Soft cost guardrail: check BEFORE spending more tokens. ---
    used = get_daily_usage()
    if not allow_over_budget and used >= settings.DAILY_TOKEN_BUDGET:
        logger.warning(
            "Daily token budget reached: %d / %d used. Raising BudgetWarning "
            "(call again with allow_over_budget=True to continue).",
            used,
            settings.DAILY_TOKEN_BUDGET,
        )
        raise BudgetWarning(
            f"Daily token budget of {settings.DAILY_TOKEN_BUDGET} reached "
            f"({used} used). Pass allow_over_budget=True to proceed."
        )

    tier_cfg = settings.MODEL_CONFIG[model_tier]
    provider = _get_provider(str(tier_cfg["provider"]))

    content, input_tokens, output_tokens = provider.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        model=str(tier_cfg["model"]),
        max_tokens=int(tier_cfg["max_tokens"]),
        response_schema=response_schema,
        images=images,
    )

    new_total = _record_usage(input_tokens, output_tokens)
    logger.debug(
        "%s/%s call: +%d in / +%d out tokens (daily total %d/%d).",
        tier_cfg["provider"],
        tier_cfg["model"],
        input_tokens,
        output_tokens,
        new_total,
        settings.DAILY_TOKEN_BUDGET,
    )

    # Warn (but don't raise) if THIS call pushed us over the line — the result
    # is already paid for, so we hand it back and let the next call enforce.
    if new_total >= settings.DAILY_TOKEN_BUDGET:
        logger.warning(
            "Daily token budget exceeded after this call: %d / %d.",
            new_total,
            settings.DAILY_TOKEN_BUDGET,
        )

    return content
