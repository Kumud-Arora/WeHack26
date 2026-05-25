"""
summarizer.py — Generates a human-readable summary of call analysis.

Strategy:
  1. Always produce a deterministic fallback summary from structured metrics.
  2. If GEMINI_API_KEY is set, attempt to upgrade the summary via Gemini;
     fall back gracefully to the deterministic version on any error.
"""

from __future__ import annotations

import os
from typing import Any


# ── Deterministic fallback ────────────────────────────────────────────────────

def _fallback_summary(metrics: dict[str, Any], raw: dict[str, Any]) -> str:
    """Build a deterministic plain-English summary from structured metrics."""
    compliance  = metrics.get("compliance_level", "low")
    pushback    = metrics.get("pushback_level", "low")
    intent      = metrics.get("spending_intent", "unclear")
    category    = metrics.get("purchase_category", "unknown")
    speed       = metrics.get("decision_speed", "undecided")
    persuade    = metrics.get("persuadability_score", 0.0)
    persona     = metrics.get("persona_effectiveness", {}).get("persona", "neutral")
    impulse_flg = metrics.get("impulse_flag", False)

    parts: list[str] = []

    # Compliance
    if compliance == "high":
        parts.append("User was highly receptive to suggestions.")
    elif compliance == "medium":
        parts.append("User showed moderate agreement.")
    else:
        parts.append("User showed little compliance during this call.")

    # Pushback
    if pushback == "high":
        parts.append("There was significant resistance throughout the conversation.")
    elif pushback == "medium":
        parts.append("User pushed back on some recommendations.")

    # Intent and category
    if intent == "impulse":
        parts.append(
            f"Spending appeared impulsive"
            + (f" in the {category} category." if category != "unknown" else ".")
        )
    elif intent == "planned":
        parts.append(
            f"This was a planned purchase"
            + (f" ({category})." if category != "unknown" else ".")
        )
    elif intent == "essential":
        parts.append(
            f"Purchase was identified as an essential expense"
            + (f" ({category})." if category != "unknown" else ".")
        )
    else:
        parts.append("Spending intent was unclear.")

    # Decision speed
    if speed == "fast":
        parts.append("User reached a decision quickly.")
    elif speed == "slow":
        parts.append("User took a long time to decide.")
    elif speed == "undecided":
        parts.append("No clear decision was made by the end of the call.")

    # Persuadability
    if persuade >= 0.65:
        parts.append(f"User is highly persuadable with the '{persona}' persona.")
    elif persuade <= 0.25:
        parts.append(f"User was difficult to persuade even with the '{persona}' persona.")

    # Impulse flag
    if impulse_flg and intent != "impulse":
        parts.append("Impulse buying signals were detected.")

    return " ".join(parts)


# ── Gemini integration (optional) ─────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Attempt a Gemini API call; return its text response.
    Raises on failure so the caller can fall back.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model    = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini call failed: {exc}") from exc


# ── Public summariser class ───────────────────────────────────────────────────

class Summarizer:
    """
    Produce a human-readable summary of call analysis output.

    Usage::

        summarizer = Summarizer()
        text = summarizer.summarize(metrics, raw_features)
    """

    def summarize(
        self,
        metrics: dict[str, Any],
        raw_features: dict[str, Any],
        conversation_id: str = "",
    ) -> str:
        """
        Return a summary string. Uses Gemini if available, deterministic fallback otherwise.
        """
        deterministic = _fallback_summary(metrics, raw_features)

        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            return deterministic

        prompt = self._build_prompt(metrics, raw_features, deterministic, conversation_id)
        try:
            return call_gemini(prompt)
        except Exception:
            return deterministic

    @staticmethod
    def _build_prompt(
        metrics: dict[str, Any],
        raw: dict[str, Any],
        fallback: str,
        conversation_id: str,
    ) -> str:
        return (
            f"You are a financial coaching assistant analysing a voice call transcript.\n"
            f"Conversation ID: {conversation_id or 'unknown'}\n\n"
            f"Structured metrics:\n{metrics}\n\n"
            f"Raw features:\n{raw}\n\n"
            f"Deterministic baseline summary:\n{fallback}\n\n"
            f"Write a 2-3 sentence plain-English coaching insight that builds on the "
            f"baseline. Be empathetic and actionable. Do not repeat the baseline verbatim."
        )
