"""
behavior_extractor.py — Deterministic signal extraction from call transcripts.

Extracts 9 raw behavioral signals from a list of conversation turns:
  1. compliance_score      — weighted phrase hit score (0.0–1.0+)
  2. pushback_count        — count of resistance/pushback phrases
  3. hesitation_count      — count of filler/uncertainty phrases
  4. impulse_count         — count of impulse-buying phrases
  5. decision_turn         — first turn index where a decision phrase appears (-1 if none)
  6. spending_type         — "impulse" | "planned" | "essential" | "unclear"
  7. purchase_category     — inferred category string or "unknown"
  8. total_turns           — number of turns in the conversation
  9. persona_used          — persona label from the conversation metadata
"""

from __future__ import annotations

import re
from typing import Any

from .config import (
    COMPLIANCE_MAP,
    COMPLIANCE_PHRASES,
    DECISION_PHRASES,
    ESSENTIAL_INTENT_PHRASES,
    HESITATION_PHRASES,
    IMPULSE_INTENT_PHRASES,
    IMPULSE_PHRASES,
    PLANNED_INTENT_PHRASES,
    PURCHASE_CATEGORY_KEYWORDS,
    PUSHBACK_PHRASES,
    VALID_PERSONAS,
)

# Pre-compile phrase → regex map for each phrase list
def _build_phrase_patterns(phrases: list[str]) -> list[tuple[str, re.Pattern]]:
    return [
        (p, re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE))
        for p in phrases
    ]

_COMPLIANCE_PATTERNS  = _build_phrase_patterns(COMPLIANCE_PHRASES)
_PUSHBACK_PATTERNS    = _build_phrase_patterns(PUSHBACK_PHRASES)
_HESITATION_PATTERNS  = _build_phrase_patterns(HESITATION_PHRASES)
_IMPULSE_PATTERNS     = _build_phrase_patterns(IMPULSE_PHRASES)
_DECISION_PATTERNS    = _build_phrase_patterns(DECISION_PHRASES)

_IMPULSE_INTENT_PATTERNS  = _build_phrase_patterns(IMPULSE_INTENT_PHRASES)
_PLANNED_INTENT_PATTERNS  = _build_phrase_patterns(PLANNED_INTENT_PHRASES)
_ESSENTIAL_INTENT_PATTERNS = _build_phrase_patterns(ESSENTIAL_INTENT_PHRASES)

_CATEGORY_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    cat: _build_phrase_patterns(kws)
    for cat, kws in PURCHASE_CATEGORY_KEYWORDS.items()
}


class BehaviorExtractor:
    """
    Extract raw behavioral features from a conversation.

    Usage::

        extractor = BehaviorExtractor()
        features = extractor.extract(conversation)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """
        Return a dict of raw behavioral signals.

        Parameters
        ----------
        conversation : dict
            Must contain "turns" (list of dicts with "speaker" and "text")
            and optionally "persona" (str).
        """
        turns        = conversation.get("turns", [])
        persona_raw  = conversation.get("persona", "neutral")
        persona      = persona_raw if persona_raw in VALID_PERSONAS else "neutral"

        # Collect only the user side for most signals
        user_texts   = [
            t.get("text", "")
            for t in turns
            if str(t.get("speaker", "")).lower() == "user"
        ]
        full_text    = " ".join(user_texts)

        return {
            "compliance_score":  self._compliance_score(user_texts),
            "pushback_count":    self._count_hits(full_text, _PUSHBACK_PATTERNS),
            "hesitation_count":  self._count_hits(full_text, _HESITATION_PATTERNS),
            "impulse_count":     self._count_hits(full_text, _IMPULSE_PATTERNS),
            "decision_turn":     self._decision_turn(turns),
            "spending_type":     self._spending_type(full_text),
            "purchase_category": self._purchase_category(full_text),
            "total_turns":       len(turns),
            "persona_used":      persona,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _count_hits(
        text: str,
        patterns: list[tuple[str, re.Pattern]],
    ) -> int:
        """Count the number of distinct phrase matches in text."""
        return sum(1 for _, pat in patterns if pat.search(text))

    def _compliance_score(self, user_texts: list[str]) -> float:
        """
        Sum COMPLIANCE_MAP weights for each matched phrase across all user turns.
        Capped at 5.0.
        """
        total = 0.0
        combined = " ".join(user_texts)
        for phrase, pat in _COMPLIANCE_PATTERNS:
            if pat.search(combined):
                total += COMPLIANCE_MAP.get(phrase, 1.0)
        return min(total, 5.0)

    @staticmethod
    def _decision_turn(turns: list[dict]) -> int:
        """
        Return the index (0-based) of the first user turn in which any
        DECISION_PHRASES is found.  Returns -1 if none found.
        """
        user_turn_idx = 0
        for turn in turns:
            if str(turn.get("speaker", "")).lower() != "user":
                continue
            text = turn.get("text", "")
            for _, pat in _DECISION_PATTERNS:
                if pat.search(text):
                    return user_turn_idx
            user_turn_idx += 1
        return -1

    def _spending_type(self, text: str) -> str:
        """
        Classify spending intent based on keyword matches.
        Priority: essential > planned > impulse > unclear
        """
        essential = self._count_hits(text, _ESSENTIAL_INTENT_PATTERNS)
        planned   = self._count_hits(text, _PLANNED_INTENT_PATTERNS)
        impulse   = self._count_hits(text, _IMPULSE_INTENT_PATTERNS)

        if essential >= max(planned, impulse) and essential > 0:
            return "essential"
        if planned > impulse and planned > 0:
            return "planned"
        if impulse > 0:
            return "impulse"
        return "unclear"

    def _purchase_category(self, text: str) -> str:
        """
        Return the category with the most keyword hits. "unknown" if none.
        """
        best_cat   = "unknown"
        best_count = 0
        for cat, patterns in _CATEGORY_PATTERNS.items():
            count = self._count_hits(text, patterns)
            if count > best_count:
                best_count = count
                best_cat   = cat
        return best_cat
