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
    max_tokens: int = 2048,
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
        Soft limit on response length (2048 allows full responses).

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
            f"RESPOND LIKE YOU'RE A REAL FRIEND GIVING A REALITY CHECK — super casual, conversational, but honest.\n"
            f"IMPORTANT: Use the financial context above (budget, spending, account info) to actually answer their question.\n"
            f"Reference their specific numbers and habits. Don't say you don't know — you have their data!\n\n"
            f"FOR LUXURY PURCHASES (iPads, laptops, expensive gadgets, sneakers, designer stuff):\n"
            f"- Default answer is NO. Be nice about it but firm.\n"
            f"- Reference how much they're already spending and their budget.\n"
            f"- Say something like: 'Nah[pause_500ms], not right now.' or 'Girl/Dude, look at your numbers. That's a no.'\n"
            f"- Show them they're already at or over their limit.\n"
            f"- Be supportive but real: 'I get it, but you gotta focus on what you [emphasis]actually[/emphasis] need.'\n\n"
            f"FOR ESSENTIAL/NECESSARY PURCHASES:\n"
            f"- If it's genuinely needed, check if they have room in their budget.\n"
            f"- Be encouraging if they can afford it.\n\n"
            f"General guidelines:\n"
            f"- Keep it flowing naturally. 5-6 sentences is great.\n"
            f"- Use contractions (I'm, you're, don't, can't, gotta, wanna)\n"
            f"- Interrupt yourself with 'like', 'literally', 'honestly', 'ngl'\n"
            f"- Drop casual filler: 'Well...', 'Umm...', 'Okay so...'\n"
            f"- Be sassy, witty, but still helpful and supportive\n"
            f"- Sound like a real friend who cares about you\n\n"
            f"ADD EMOTION TAGS FOR NATURAL SPEECH RHYTHM:\n"
            f"- [pause_500ms] after filler words\n"
            f"- [emphasis]key words[/emphasis] when making a point\n"
            f"- [pitch_high][rate_slow]questions[/rate_slow][/pitch_high] to sound naturally inquisitive\n"
            f"\n"
            f"Examples of good responses:\n"
            f"'Nah[pause_500ms], not right now. You're already at 90 bucks on food this week. An iPad's not happening.'\n"
            f"'Okay so[pause_500ms], you got a [emphasis]hundred[/emphasis] dollar budget and you've already spent most of it. Nope, no laptop.'\n"
            f"'I get it, they're cool. But girl, look at your numbers. That's not in the cards right now.'\n"
            f"\n"
            f"Be real, be nice, but give them the truth."
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
