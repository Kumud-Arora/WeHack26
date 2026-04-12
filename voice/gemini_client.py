"""
gemini_client.py — Thin wrapper around Google's Generative AI SDK.

Provides a single ask() function that:
  • Takes a system prompt and user message
  • Returns a short voice-friendly text reply
  • Falls back gracefully if the API key is missing or the call fails

The model is lazily initialised once and cached for the process lifetime.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Fallback strings ──────────────────────────────────────────────────────────

_FALLBACK_NO_KEY = (
    "My AI service is not configured yet. "
    "Please ask the administrator to set the Gemini API key."
)
_FALLBACK_ERROR = (
    "I'm having some trouble right now. "
    "Please try asking your question again."
)

# ── Lazy model cache ──────────────────────────────────────────────────────────

_model_cache: dict[str, Any] = {}


def _get_model():
    """
    Lazily create and cache the GenerativeModel.
    Returns (model, error_string_or_None).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, _FALLBACK_NO_KEY

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    if model_name in _model_cache:
        return _model_cache[model_name], None

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _model_cache[model_name] = model
        logger.info("Gemini model initialised: %s", model_name)
        return model, None

    except ImportError:
        msg = "google-generativeai package not installed. Run: pip install google-generativeai"
        logger.error(msg)
        return None, msg

    except Exception as exc:
        logger.error("Gemini model init failed: %s", exc)
        return None, _FALLBACK_ERROR


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 500,
) -> str:
    """
    Send a single-turn question to Gemini with the given system context.

    Parameters
    ----------
    system_prompt : str
        Financial context + persona instructions (built by context_builder).
    user_message : str
        The user's question / speech transcript from Twilio.
    max_tokens : int
        Soft limit on response length (150 ≈ 1-2 spoken sentences).

    Returns
    -------
    str
        A short, voice-friendly reply. Never raises — always returns a string.
    """
    model, err = _get_model()
    if err:
        logger.warning("Gemini unavailable: %s", err)
        return err

    try:
        import google.generativeai as genai  # type: ignore

        full_prompt = (
            f"{system_prompt}\n\n"
            f"User says: {user_message}\n\n"
            f"Reply in natural spoken sentences. Simple questions: 1-2 sentences. "
            f"Complex financial questions: up to 5-6 sentences, but always finish your thought. "
            f"No lists, no markdown, no headers, no bullet points."
        )

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,  # 500 allows ~5-6 spoken sentences
                temperature=0.7,
            ),
        )

        text = response.text.strip()

        # Sanitise for TwiML <Say>:
        # 1. Strip markdown that leaks through
        text = text.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        # 2. Collapse newlines/tabs into a single space so <Say> speaks cleanly
        text = " ".join(text.split())

        logger.debug("Gemini reply: %s", text)
        return text

    except Exception as exc:
        logger.error("Gemini call failed: %s", exc, exc_info=True)
        return _FALLBACK_ERROR


def invalidate_cache() -> None:
    """Clear the model cache (useful after config changes in tests)."""
    _model_cache.clear()
