"""
metrics_calculator.py — Derives structured behavioral metrics from raw features.

Input:  raw_features dict produced by BehaviorExtractor.extract()
Output: metrics dict with:
    - compliance_level       str  "high" | "medium" | "low"
    - pushback_level         str  "high" | "medium" | "low"
    - spending_intent        str  "impulse" | "planned" | "essential" | "unclear"
    - purchase_category      str  (passed through from extractor)
    - confidence_score       float  0.0–1.0
    - persuadability_score   float  0.0–1.0
    - persona_effectiveness  dict  {pushback_rate, persuadability}
    - decision_speed         str  "fast" | "medium" | "slow" | "undecided"
    - impulse_flag           bool
"""

from __future__ import annotations

from typing import Any

from .config import CONFIDENCE_WEIGHTS, VALID_PERSONAS


class MetricsCalculator:
    """
    Compute structured metrics from raw behavioral features.

    Usage::

        calc    = MetricsCalculator()
        metrics = calc.calculate(raw_features)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(self, raw: dict[str, Any]) -> dict[str, Any]:
        compliance  = raw.get("compliance_score", 0.0)
        pushback    = raw.get("pushback_count", 0)
        hesitation  = raw.get("hesitation_count", 0)
        impulse     = raw.get("impulse_count", 0)
        decision    = raw.get("decision_turn", -1)
        total_turns = max(raw.get("total_turns", 1), 1)
        persona     = raw.get("persona_used", "neutral")
        spending    = raw.get("spending_type", "unclear")
        category    = raw.get("purchase_category", "unknown")

        if spending not in ("impulse", "planned", "essential", "unclear"):
            spending = "unclear"

        persona = persona if persona in VALID_PERSONAS else "neutral"

        conf         = self._confidence_score(total_turns, decision, hesitation, compliance, impulse)
        persuadability = self._persuadability_score(compliance, pushback, hesitation)

        return {
            "compliance_level":      self._level(compliance, lo=1.0, hi=2.5),
            "pushback_level":        self._pushback_level(pushback),
            "spending_intent":       spending,
            "purchase_category":     category,
            "confidence_score":      conf,
            "persuadability_score":  persuadability,
            "persona_effectiveness": self._persona_effectiveness(pushback, total_turns, persuadability, persona),
            "decision_speed":        self._decision_speed(decision, total_turns),
            "impulse_flag":          impulse >= 2 or spending == "impulse",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))

    def _level(self, value: float, lo: float, hi: float) -> str:
        if value >= hi:
            return "high"
        if value >= lo:
            return "medium"
        return "low"

    def _pushback_level(self, pushback: int) -> str:
        if pushback >= 4:
            return "high"
        if pushback >= 2:
            return "medium"
        return "low"

    def _confidence_score(
        self,
        total_turns: int,
        decision_turn: int,
        hesitation: int,
        compliance: float,
        impulse: int,
    ) -> float:
        w = CONFIDENCE_WEIGHTS

        # Contribution from turn count
        turn_contrib = self._clamp(total_turns * w["per_turn"], 0.0, w["turn_cap"])

        # Outcome clarity
        if decision_turn == -1:
            outcome_contrib = 0.0
        elif decision_turn == 0:
            outcome_contrib = w["outcome_very_clear"]
        else:
            outcome_contrib = w["outcome_clear"]

        # Hesitation penalty (each hesitation phrase reduces confidence)
        penalty = self._clamp(hesitation * w["hesitation_penalty"], 0.0, 0.20)

        # Signal bonus if compliance or impulse present
        bonus = w["signal_bonus"] if (compliance > 0.5 or impulse > 0) else 0.0

        raw = turn_contrib + outcome_contrib - penalty + bonus
        return round(self._clamp(raw), 3)

    def _persuadability_score(
        self,
        compliance: float,
        pushback: int,
        hesitation: int,
    ) -> float:
        """
        High compliance + low pushback = highly persuadable.
        Hesitation is a slight bonus (undecided → persuadable).
        """
        base    = self._clamp(compliance / 5.0)          # 0–1 normalised
        penalty = self._clamp(pushback * 0.12, 0.0, 0.60)
        bonus   = self._clamp(hesitation * 0.03, 0.0, 0.15)
        return round(self._clamp(base - penalty + bonus), 3)

    @staticmethod
    def _persona_effectiveness(
        pushback: int,
        total_turns: int,
        persuadability: float,
        persona: str,
    ) -> dict[str, Any]:
        pushback_rate = round(pushback / max(total_turns, 1), 3)
        return {
            "persona":         persona,
            "pushback_rate":   pushback_rate,
            "persuadability":  persuadability,
        }

    @staticmethod
    def _decision_speed(decision_turn: int, total_turns: int) -> str:
        if decision_turn == -1:
            return "undecided"
        user_turns = max(total_turns // 2, 1)   # rough estimate of user turn count
        ratio = decision_turn / max(user_turns, 1)
        if ratio <= 0.25:
            return "fast"
        if ratio <= 0.60:
            return "medium"
        return "slow"
