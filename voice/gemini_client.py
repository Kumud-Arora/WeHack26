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
    max_tokens: int = 1024,
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
        Soft limit on response length (1024 ≈ 10-12 spoken sentences).

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
            f"Reply like you're a confident, sassy friend giving financial advice on the phone. "
            f"Be witty, fun, and a little cheeky - but still helpful. "
            f"Use contractions (I'm, you're, it's, don't, can't, etc). "
            f"Drop some attitude and personality - think sarcasm, eye-roll energy, playful confidence. "
            f"Keep replies to 1-3 sentences max - punchy and quick like real banter. "
            f"\n"
            f"IMPORTANT: ADD EMOTION TAGS TO YOUR RESPONSE FOR NATURAL, EXPRESSIVE SPEECH:\n"
            f"- Use [pause_500ms] after thinking words like 'Hmm,', 'Well,', 'Actually,' to add natural pauses\n"
            f"- Wrap [emphasis]important words[/emphasis] to stress them (e.g., 'That's [emphasis]literally[/emphasis] a waste')\n"
            f"- For questions, wrap the whole thing: [pitch_high][rate_slow]Is that even a question?[/rate_slow][/pitch_high]\n"
            f"- Use [pause_200ms] between sentences for natural timing\n"
            f"Example responses:\n"
            f"'Well[pause_500ms], that's [emphasis]literally[/emphasis] what got you into this mess.'\n"
            f"'[pitch_high][rate_slow]Do you even look at your bank statement?[/rate_slow][/pitch_high]'\n"
            f"\n"
            f"Use simple, modern words. Sound like you actually care but aren't taking it too seriously. "
            f"No lists, no markdown, no formal language. Just confident, sassy speech WITH emotion tags naturally woven in."
        )

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.9,  # Higher for more personality and sass
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
