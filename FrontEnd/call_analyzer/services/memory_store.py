"""
memory_store.py — Persistent per-user behavioral memory.

Memory schema (stored as JSON):
{
  "user_id": str,
  "total_calls": int,
  "avg_compliance": float,
  "avg_pushback": float,
  "avg_hesitation": float,
  "avg_confidence": float,
  "avg_persuadability": float,
  "impulse_frequency": float,          # Bernoulli rolling mean (0–1)
  "most_common_purchase_category": str,
  "category_counts": { cat: int, ... },
  "risk_tendency": "high" | "medium" | "low",
  "preferred_persona": str | null,     # null until >= 2 calls
  "persona_stats": {
    persona: { "calls": int, "avg_compliance": float, "avg_persuadability": float }
  },
  "history": [
    { "call_index": int, "compliance": float, "pushback": int,
      "spending_type": str, "purchase_category": str,
      "confidence": float, "persuadability": float, "impulse_flag": bool }
  ]
}
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import (
    MEMORY_FILE_PATH,
    PREFERRED_PERSONA_MIN_CALLS,
    RISK_THRESHOLDS,
    VALID_PERSONAS,
)

_DEFAULT_MEMORY: dict[str, Any] = {
    "user_id":                     "",
    "total_calls":                 0,
    "avg_compliance":              0.0,
    "avg_pushback":                0.0,
    "avg_hesitation":              0.0,
    "avg_confidence":              0.0,
    "avg_persuadability":          0.0,
    "impulse_frequency":           0.0,
    "most_common_purchase_category": "unknown",
    "category_counts":             {},
    "risk_tendency":               "low",
    "preferred_persona":           None,
    "persona_stats":               {},
    "history":                     [],
}


class MemoryStore:
    """
    Load, update, and persist per-user behavioral memory.

    Parameters
    ----------
    memory_file : Path | str | None
        Path to the JSON memory file.  Defaults to config.MEMORY_FILE_PATH.
        Pass a temp path in tests to avoid touching the real data file.
    """

    def __init__(self, memory_file: Path | str | None = None) -> None:
        self._path = Path(memory_file) if memory_file else MEMORY_FILE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, user_id: str) -> dict[str, Any]:
        """Load memory for *user_id*.  Returns a fresh default if not found."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fp:
                    all_memory: dict = json.load(fp)
                if user_id in all_memory:
                    merged = copy.deepcopy(_DEFAULT_MEMORY)
                    merged.update(copy.deepcopy(all_memory[user_id]))
                    return merged
            except (json.JSONDecodeError, KeyError):
                pass

        mem = copy.deepcopy(_DEFAULT_MEMORY)
        mem["user_id"] = user_id
        return mem

    def save(self, user_id: str, memory: dict[str, Any]) -> None:
        """Persist *memory* for *user_id* into the shared JSON file."""
        all_memory: dict = {}
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fp:
                    all_memory = json.load(fp)
            except (json.JSONDecodeError, ValueError):
                all_memory = {}

        all_memory[user_id] = memory

        with open(self._path, "w", encoding="utf-8") as fp:
            json.dump(all_memory, fp, indent=2, ensure_ascii=False)

    def update(
        self,
        user_id: str,
        raw_features: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Load existing memory, apply new call data, save, and return updated memory.
        """
        mem = self.load(user_id)
        n   = mem["total_calls"]   # calls BEFORE this one

        # Rolling averages for numeric signals
        mem["avg_compliance"]    = self._rolling_avg(mem["avg_compliance"],    raw_features.get("compliance_score", 0.0), n)
        mem["avg_pushback"]      = self._rolling_avg(mem["avg_pushback"],      raw_features.get("pushback_count", 0),     n)
        mem["avg_hesitation"]    = self._rolling_avg(mem["avg_hesitation"],    raw_features.get("hesitation_count", 0),   n)
        mem["avg_confidence"]    = self._rolling_avg(mem["avg_confidence"],    metrics.get("confidence_score", 0.0),      n)
        mem["avg_persuadability"]= self._rolling_avg(mem["avg_persuadability"],metrics.get("persuadability_score", 0.0),  n)

        # Impulse frequency (Bernoulli: was this call impulsive?)
        impulse_val = 1.0 if metrics.get("impulse_flag", False) else 0.0
        mem["impulse_frequency"] = self._rolling_avg(mem["impulse_frequency"], impulse_val, n)

        # Increment call count AFTER rolling avg (n was old count)
        mem["total_calls"] = n + 1

        # Purchase category tracking
        cat = metrics.get("purchase_category", "unknown")
        counts: dict[str, int] = mem.get("category_counts") or {}
        counts[cat] = counts.get(cat, 0) + 1
        mem["category_counts"] = counts
        mem["most_common_purchase_category"] = max(counts, key=lambda k: counts[k])

        # Persona stats
        persona = raw_features.get("persona_used", "neutral")
        if persona not in VALID_PERSONAS:
            persona = "neutral"
        stats: dict = mem.get("persona_stats") or {}
        if persona not in stats:
            stats[persona] = {"calls": 0, "avg_compliance": 0.0, "avg_persuadability": 0.0}
        p      = stats[persona]
        p_n    = p["calls"]
        p["avg_compliance"]    = self._rolling_avg(p["avg_compliance"],    raw_features.get("compliance_score", 0.0),  p_n)
        p["avg_persuadability"]= self._rolling_avg(p["avg_persuadability"],metrics.get("persuadability_score", 0.0),   p_n)
        p["calls"]             = p_n + 1
        mem["persona_stats"]   = stats

        # Preferred persona (only commit if >= PREFERRED_PERSONA_MIN_CALLS total)
        if mem["total_calls"] >= PREFERRED_PERSONA_MIN_CALLS:
            mem["preferred_persona"] = self._best_persona(stats)
        else:
            mem["preferred_persona"] = None

        # Risk tendency (multi-factor)
        mem["risk_tendency"] = self._compute_risk_tendency(
            mem["avg_pushback"], mem["avg_compliance"]
        )

        # History entry
        mem["history"].append({
            "call_index":       mem["total_calls"],
            "compliance":       raw_features.get("compliance_score", 0.0),
            "pushback":         raw_features.get("pushback_count", 0),
            "spending_type":    metrics.get("spending_intent", "unclear"),
            "purchase_category": cat,
            "confidence":       metrics.get("confidence_score", 0.0),
            "persuadability":   metrics.get("persuadability_score", 0.0),
            "impulse_flag":     metrics.get("impulse_flag", False),
        })

        self.save(user_id, mem)
        return mem

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rolling_avg(current: float, new_val: float, n: int) -> float:
        """Incremental running mean: (current * n + new_val) / (n + 1)."""
        return round((current * n + new_val) / (n + 1), 4)

    @staticmethod
    def _compute_risk_tendency(avg_pushback: float, avg_compliance: float) -> str:
        t = RISK_THRESHOLDS
        if avg_pushback >= t["high_pushback_min"] or avg_compliance <= t["low_compliance_max"]:
            return "high"
        if avg_pushback >= t["medium_pushback_min"] or avg_compliance <= t["medium_compliance_max"]:
            return "medium"
        return "low"

    @staticmethod
    def _best_persona(persona_stats: dict) -> str | None:
        """Return persona with highest avg_persuadability (needs ≥1 call)."""
        candidates = {
            p: s["avg_persuadability"]
            for p, s in persona_stats.items()
            if s.get("calls", 0) >= 1
        }
        if not candidates:
            return None
        return max(candidates, key=lambda k: candidates[k])
