"""
test_cases.py — Unit and integration tests for call_analyzer.

Run from project root:
    python call_analyzer/tests/test_cases.py
    python call_analyzer/tests/test_cases.py --demo

Or with pytest:
    python -m pytest call_analyzer/tests/test_cases.py -v

5 fixture scenarios
───────────────────
1. compliant_hesitant_mean   — user agrees but keeps second-guessing; mean outcome
2. resistant_impulse         — strong pushback + impulsive buy anyway; bad outcome
3. mom_success               — encouraging persona converts a reluctant parent; success
4. dad_fail                  — strict persona bounces off; no decision reached
5. finance_coach_cheaper     — user asks about cheaper option; essential spend; planned
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make sure package is importable from project root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from call_analyzer.main.postprocess import process_conversation, process_conversations_batch
from call_analyzer.services.behavior_extractor import BehaviorExtractor
from call_analyzer.services.memory_store import MemoryStore
from call_analyzer.services.metrics_calculator import MetricsCalculator
from call_analyzer.services.summarizer import _fallback_summary

# ── Fixtures ──────────────────────────────────────────────────────────────────

COMPLIANT_HESITANT_MEAN = {
    "id": "conv_001",
    "persona": "encouraging",
    "turns": [
        {"speaker": "assistant", "text": "Hi! How can I help with your budget today?"},
        {"speaker": "user",      "text": "Um, well, I spent a lot on food this week I guess."},
        {"speaker": "assistant", "text": "Let's look at that. You spent $180 on dining out."},
        {"speaker": "user",      "text": "Yeah, okay, that makes sense. Maybe I should cut back."},
        {"speaker": "assistant", "text": "Great idea! Try cooking at home 3 nights a week."},
        {"speaker": "user",      "text": "I suppose, I mean, sure I'll try that."},
        {"speaker": "assistant", "text": "You're doing great! Any other concerns?"},
        {"speaker": "user",      "text": "Not really, I think I'm good for now. Alright, will do."},
    ],
}

RESISTANT_IMPULSE = {
    "id": "conv_002",
    "persona": "strict",
    "turns": [
        {"speaker": "assistant", "text": "Let's review your recent purchases."},
        {"speaker": "user",      "text": "But I need all of it. It was on sale at Amazon, couldn't resist."},
        {"speaker": "assistant", "text": "You spent $340 on electronics this month."},
        {"speaker": "user",      "text": "That's too much? No way, my Amazon order was just things I needed."},
        {"speaker": "assistant", "text": "Consider returning the items you haven't opened."},
        {"speaker": "user",      "text": "I won't do that. I deserve it, treat myself sometimes."},
        {"speaker": "assistant", "text": "Your savings goal will be delayed by 2 months."},
        {"speaker": "user",      "text": "I don't think that's right. However, I'll reconsider... maybe."},
    ],
}

MOM_SUCCESS = {
    "id": "conv_003",
    "persona": "encouraging",
    "turns": [
        {"speaker": "assistant", "text": "Hi! I noticed your grocery bill went up this month."},
        {"speaker": "user",      "text": "I know, I had to stock up, we needed groceries."},
        {"speaker": "assistant", "text": "That's understandable! You could try meal planning."},
        {"speaker": "user",      "text": "Good idea! I'll do that. Let's start meal planning, sounds good."},
        {"speaker": "assistant", "text": "Would you like help setting a weekly grocery budget?"},
        {"speaker": "user",      "text": "Absolutely! That would be perfect. Let's set it up."},
    ],
}

DAD_FAIL = {
    "id": "conv_004",
    "persona": "strict",
    "turns": [
        {"speaker": "assistant", "text": "Your gas spending is 40% above the monthly average."},
        {"speaker": "user",      "text": "But I commute every day, gas is expensive, I can't help it."},
        {"speaker": "assistant", "text": "You should carpool or take transit twice a week."},
        {"speaker": "user",      "text": "That won't work. I have different hours, it's just not possible."},
        {"speaker": "assistant", "text": "Even one carpool day saves $30 a month."},
        {"speaker": "user",      "text": "I'm not sure. I doubt I'd find someone with my schedule."},
        {"speaker": "assistant", "text": "Let's set a goal to try it once this week."},
        {"speaker": "user",      "text": "Hmm, however, I really don't think so. Not convinced."},
    ],
}

FINANCE_COACH_CHEAPER = {
    "id": "conv_005",
    "persona": "friendly",
    "turns": [
        {"speaker": "assistant", "text": "I see you're considering upgrading your phone plan."},
        {"speaker": "user",      "text": "Yeah, I've been comparing my phone bill, been planning this for a while."},
        {"speaker": "assistant", "text": "Your current bill is $85/month. Mint Mobile is $30."},
        {"speaker": "user",      "text": "That makes sense! I researched the bill too. Sounds good."},
        {"speaker": "assistant", "text": "Switching saves $660 a year. Shall we walk through it?"},
        {"speaker": "user",      "text": "You're right, absolutely! I'll do it. Great idea, you've convinced me."},
    ],
}

ALL_FIXTURES = [
    COMPLIANT_HESITANT_MEAN,
    RESISTANT_IMPULSE,
    MOM_SUCCESS,
    DAD_FAIL,
    FINANCE_COACH_CHEAPER,
]


# ── Test classes ──────────────────────────────────────────────────────────────

class TestBehaviorExtractor(unittest.TestCase):
    """Tests for BehaviorExtractor.extract()"""

    def setUp(self):
        self.ex = BehaviorExtractor()

    # --- compliant_hesitant_mean ---
    def test_compliant_hesitant_compliance_nonzero(self):
        f = self.ex.extract(COMPLIANT_HESITANT_MEAN)
        self.assertGreater(f["compliance_score"], 0.0)

    def test_compliant_hesitant_hesitation_detected(self):
        f = self.ex.extract(COMPLIANT_HESITANT_MEAN)
        self.assertGreater(f["hesitation_count"], 0)

    def test_compliant_hesitant_pushback_low(self):
        f = self.ex.extract(COMPLIANT_HESITANT_MEAN)
        self.assertLessEqual(f["pushback_count"], 3)

    def test_compliant_hesitant_total_turns(self):
        f = self.ex.extract(COMPLIANT_HESITANT_MEAN)
        self.assertEqual(f["total_turns"], 8)

    def test_compliant_hesitant_persona(self):
        f = self.ex.extract(COMPLIANT_HESITANT_MEAN)
        self.assertEqual(f["persona_used"], "encouraging")

    # --- resistant_impulse ---
    def test_resistant_pushback_high(self):
        f = self.ex.extract(RESISTANT_IMPULSE)
        self.assertGreaterEqual(f["pushback_count"], 4)

    def test_resistant_impulse_count(self):
        f = self.ex.extract(RESISTANT_IMPULSE)
        self.assertGreaterEqual(f["impulse_count"], 2)

    def test_resistant_spending_type_impulse(self):
        f = self.ex.extract(RESISTANT_IMPULSE)
        self.assertEqual(f["spending_type"], "impulse")

    def test_resistant_compliance_low(self):
        f = self.ex.extract(RESISTANT_IMPULSE)
        self.assertLess(f["compliance_score"], 1.5)

    def test_resistant_purchase_category_shopping(self):
        f = self.ex.extract(RESISTANT_IMPULSE)
        self.assertEqual(f["purchase_category"], "shopping")

    # --- mom_success ---
    def test_mom_compliance_high(self):
        f = self.ex.extract(MOM_SUCCESS)
        self.assertGreaterEqual(f["compliance_score"], 3.0)

    def test_mom_spending_type_essential(self):
        f = self.ex.extract(MOM_SUCCESS)
        self.assertEqual(f["spending_type"], "essential")

    def test_mom_decision_turn_found(self):
        f = self.ex.extract(MOM_SUCCESS)
        self.assertGreaterEqual(f["decision_turn"], 0)

    def test_mom_persona_encouraging(self):
        f = self.ex.extract(MOM_SUCCESS)
        self.assertEqual(f["persona_used"], "encouraging")

    # --- dad_fail ---
    def test_dad_pushback_high(self):
        f = self.ex.extract(DAD_FAIL)
        self.assertGreaterEqual(f["pushback_count"], 3)

    def test_dad_decision_turn_not_found(self):
        f = self.ex.extract(DAD_FAIL)
        self.assertEqual(f["decision_turn"], -1)

    def test_dad_purchase_category_gas(self):
        f = self.ex.extract(DAD_FAIL)
        self.assertEqual(f["purchase_category"], "gas")

    def test_dad_persona_strict(self):
        f = self.ex.extract(DAD_FAIL)
        self.assertEqual(f["persona_used"], "strict")

    # --- finance_coach_cheaper ---
    def test_finance_coach_planned_intent(self):
        f = self.ex.extract(FINANCE_COACH_CHEAPER)
        self.assertEqual(f["spending_type"], "planned")

    def test_finance_coach_compliance_high(self):
        f = self.ex.extract(FINANCE_COACH_CHEAPER)
        self.assertGreaterEqual(f["compliance_score"], 3.0)

    def test_finance_coach_decision_found(self):
        f = self.ex.extract(FINANCE_COACH_CHEAPER)
        self.assertGreaterEqual(f["decision_turn"], 0)

    def test_finance_coach_category_bills(self):
        f = self.ex.extract(FINANCE_COACH_CHEAPER)
        self.assertEqual(f["purchase_category"], "bills")

    def test_invalid_persona_defaults_to_neutral(self):
        conv = dict(COMPLIANT_HESITANT_MEAN, persona="sarcastic")
        f    = self.ex.extract(conv)
        self.assertEqual(f["persona_used"], "neutral")

    def test_no_turns_returns_zero_signals(self):
        f = self.ex.extract({"turns": []})
        self.assertEqual(f["compliance_score"], 0.0)
        self.assertEqual(f["pushback_count"], 0)
        self.assertEqual(f["total_turns"], 0)
        self.assertEqual(f["decision_turn"], -1)


class TestMetricsCalculator(unittest.TestCase):
    """Tests for MetricsCalculator.calculate()"""

    def setUp(self):
        self.ex   = BehaviorExtractor()
        self.calc = MetricsCalculator()

    def _calc(self, fixture: dict) -> dict:
        return self.calc.calculate(self.ex.extract(fixture))

    # Compliant hesitant
    def test_compliant_hesitant_persuadability_medium(self):
        m = self._calc(COMPLIANT_HESITANT_MEAN)
        self.assertGreater(m["persuadability_score"], 0.2)

    def test_compliant_hesitant_compliance_level_not_low(self):
        m = self._calc(COMPLIANT_HESITANT_MEAN)
        self.assertIn(m["compliance_level"], ("medium", "high"))

    def test_compliant_hesitant_spending_intent_preserved(self):
        m = self._calc(COMPLIANT_HESITANT_MEAN)
        self.assertIn(m["spending_intent"], ("impulse", "planned", "essential", "unclear"))

    # Resistant impulse
    def test_resistant_impulse_flag_true(self):
        m = self._calc(RESISTANT_IMPULSE)
        self.assertTrue(m["impulse_flag"])

    def test_resistant_pushback_level_high(self):
        m = self._calc(RESISTANT_IMPULSE)
        self.assertEqual(m["pushback_level"], "high")

    def test_resistant_persuadability_low(self):
        m = self._calc(RESISTANT_IMPULSE)
        self.assertLess(m["persuadability_score"], 0.5)

    def test_resistant_decision_undecided_or_slow(self):
        m = self._calc(RESISTANT_IMPULSE)
        # User says "I'll reconsider... maybe" in last turn — decision may or may not be found
        self.assertIn(m["decision_speed"], ("fast", "medium", "slow", "undecided"))

    # Mom success
    def test_mom_confidence_score_positive(self):
        m = self._calc(MOM_SUCCESS)
        self.assertGreater(m["confidence_score"], 0.0)

    def test_mom_impulse_flag_false(self):
        m = self._calc(MOM_SUCCESS)
        self.assertFalse(m["impulse_flag"])

    def test_mom_decision_speed_fast_or_medium(self):
        m = self._calc(MOM_SUCCESS)
        self.assertIn(m["decision_speed"], ("fast", "medium"))

    def test_mom_persuadability_high(self):
        m = self._calc(MOM_SUCCESS)
        self.assertGreater(m["persuadability_score"], 0.4)

    # Dad fail
    def test_dad_decision_speed_undecided(self):
        m = self._calc(DAD_FAIL)
        self.assertEqual(m["decision_speed"], "undecided")

    def test_dad_pushback_level_high(self):
        m = self._calc(DAD_FAIL)
        self.assertIn(m["pushback_level"], ("medium", "high"))

    def test_dad_confidence_score_range(self):
        m = self._calc(DAD_FAIL)
        self.assertGreaterEqual(m["confidence_score"], 0.0)
        self.assertLessEqual(m["confidence_score"], 1.0)

    # Finance coach
    def test_finance_coach_spending_intent_planned(self):
        m = self._calc(FINANCE_COACH_CHEAPER)
        self.assertEqual(m["spending_intent"], "planned")

    def test_finance_coach_confidence_high(self):
        m = self._calc(FINANCE_COACH_CHEAPER)
        self.assertGreater(m["confidence_score"], 0.3)

    def test_finance_coach_persona_effectiveness_has_keys(self):
        m = self._calc(FINANCE_COACH_CHEAPER)
        pe = m["persona_effectiveness"]
        self.assertIn("persona", pe)
        self.assertIn("pushback_rate", pe)
        self.assertIn("persuadability", pe)

    # Score ranges
    def test_confidence_score_always_0_to_1(self):
        for fixture in ALL_FIXTURES:
            m = self._calc(fixture)
            self.assertGreaterEqual(m["confidence_score"], 0.0, fixture["id"])
            self.assertLessEqual(m["confidence_score"], 1.0, fixture["id"])

    def test_persuadability_score_always_0_to_1(self):
        for fixture in ALL_FIXTURES:
            m = self._calc(fixture)
            self.assertGreaterEqual(m["persuadability_score"], 0.0, fixture["id"])
            self.assertLessEqual(m["persuadability_score"], 1.0, fixture["id"])

    def test_invalid_spending_type_coerced(self):
        raw = {"spending_type": "unknown_value", "total_turns": 4,
               "compliance_score": 1.0, "pushback_count": 0,
               "hesitation_count": 0, "impulse_count": 0,
               "decision_turn": -1, "persona_used": "neutral",
               "purchase_category": "unknown"}
        m = self.calc.calculate(raw)
        self.assertEqual(m["spending_intent"], "unclear")


class TestMemoryStore(unittest.TestCase):
    """Tests for MemoryStore.update() and persistence."""

    def setUp(self):
        self.ex    = BehaviorExtractor()
        self.calc  = MetricsCalculator()
        self._tmp  = tempfile.TemporaryDirectory()
        self._mem_file = Path(self._tmp.name) / "memory.json"
        self.store = MemoryStore(self._mem_file)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fixture: dict, user_id: str = "test_user") -> dict:
        raw     = self.ex.extract(fixture)
        metrics = self.calc.calculate(raw)
        return self.store.update(user_id, raw, metrics)

    def test_total_calls_increments(self):
        m1 = self._run(MOM_SUCCESS)
        self.assertEqual(m1["total_calls"], 1)
        m2 = self._run(MOM_SUCCESS)
        self.assertEqual(m2["total_calls"], 2)

    def test_avg_compliance_updates(self):
        self._run(COMPLIANT_HESITANT_MEAN)
        mem2 = self._run(MOM_SUCCESS)
        self.assertGreater(mem2["avg_compliance"], 0.0)

    def test_preferred_persona_none_after_one_call(self):
        m = self._run(MOM_SUCCESS)
        self.assertIsNone(m["preferred_persona"])

    def test_preferred_persona_set_after_two_calls(self):
        self._run(MOM_SUCCESS)
        m2 = self._run(MOM_SUCCESS)
        self.assertIsNotNone(m2["preferred_persona"])

    def test_risk_tendency_high_for_resistant(self):
        self._run(RESISTANT_IMPULSE)
        self._run(RESISTANT_IMPULSE)
        m3 = self._run(RESISTANT_IMPULSE)
        self.assertIn(m3["risk_tendency"], ("high", "medium"))

    def test_risk_tendency_low_for_compliant(self):
        self._run(MOM_SUCCESS)
        m2 = self._run(FINANCE_COACH_CHEAPER)
        self.assertIn(m2["risk_tendency"], ("low", "medium"))

    def test_most_common_category_tracked(self):
        self._run(RESISTANT_IMPULSE)   # shopping
        self._run(RESISTANT_IMPULSE)   # shopping
        self._run(MOM_SUCCESS)         # groceries
        m = self._run(RESISTANT_IMPULSE)  # shopping
        self.assertEqual(m["most_common_purchase_category"], "shopping")

    def test_impulse_frequency_nonzero_after_impulse_call(self):
        m = self._run(RESISTANT_IMPULSE)
        self.assertGreater(m["impulse_frequency"], 0.0)

    def test_impulse_frequency_zero_after_compliant_calls(self):
        self._run(MOM_SUCCESS)
        m = self._run(FINANCE_COACH_CHEAPER)
        self.assertAlmostEqual(m["impulse_frequency"], 0.0, places=2)

    def test_history_appended(self):
        self._run(MOM_SUCCESS)
        m = self._run(DAD_FAIL)
        self.assertEqual(len(m["history"]), 2)

    def test_history_entry_has_required_keys(self):
        m = self._run(COMPLIANT_HESITANT_MEAN)
        entry = m["history"][0]
        for key in ("call_index", "compliance", "pushback", "spending_type",
                    "purchase_category", "confidence", "persuadability", "impulse_flag"):
            self.assertIn(key, entry)

    def test_persona_stats_recorded(self):
        self._run(MOM_SUCCESS)         # encouraging
        m = self._run(DAD_FAIL)        # strict
        self.assertIn("encouraging", m["persona_stats"])
        self.assertIn("strict", m["persona_stats"])

    def test_persona_stats_call_count(self):
        self._run(MOM_SUCCESS)
        self._run(MOM_SUCCESS)
        m = self._run(MOM_SUCCESS)
        self.assertEqual(m["persona_stats"]["encouraging"]["calls"], 3)

    def test_memory_persists_across_store_instances(self):
        self._run(MOM_SUCCESS, user_id="persist_user")
        store2 = MemoryStore(self._mem_file)
        mem    = store2.load("persist_user")
        self.assertEqual(mem["total_calls"], 1)

    def test_unknown_user_returns_fresh_memory(self):
        mem = self.store.load("brand_new_user_xyz")
        self.assertEqual(mem["total_calls"], 0)
        self.assertEqual(mem["preferred_persona"], None)


class TestFullPipeline(unittest.TestCase):
    """Integration tests via process_conversation / process_conversations_batch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._mem = Path(self._tmp.name) / "memory.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fixture: dict, uid: str = "pipe_user") -> dict:
        return process_conversation(fixture, uid, memory_file=self._mem)

    def test_result_has_all_top_level_keys(self):
        r = self._run(MOM_SUCCESS)
        for key in ("conversation_id", "user_id", "raw_features", "metrics", "summary", "memory"):
            self.assertIn(key, r)

    def test_conversation_id_passed_through(self):
        r = self._run(MOM_SUCCESS)
        self.assertEqual(r["conversation_id"], "conv_003")

    def test_user_id_passed_through(self):
        r = self._run(MOM_SUCCESS, uid="my_user")
        self.assertEqual(r["user_id"], "my_user")

    def test_summary_is_nonempty_string(self):
        r = self._run(DAD_FAIL)
        self.assertIsInstance(r["summary"], str)
        self.assertGreater(len(r["summary"]), 10)

    def test_batch_returns_correct_length(self):
        results = process_conversations_batch(ALL_FIXTURES, "batch_user", memory_file=self._mem)
        self.assertEqual(len(results), 5)

    def test_batch_memory_accumulates(self):
        results = process_conversations_batch(ALL_FIXTURES, "accum_user", memory_file=self._mem)
        last_mem = results[-1]["memory"]
        self.assertEqual(last_mem["total_calls"], 5)

    def test_batch_ids_preserved(self):
        results = process_conversations_batch(ALL_FIXTURES, "id_user", memory_file=self._mem)
        ids = [r["conversation_id"] for r in results]
        self.assertEqual(ids, ["conv_001", "conv_002", "conv_003", "conv_004", "conv_005"])

    def test_accepts_json_string_input(self):
        r = process_conversation(json.dumps(MOM_SUCCESS), "str_user", memory_file=self._mem)
        self.assertEqual(r["conversation_id"], "conv_003")

    def test_validation_rejects_missing_turns(self):
        with self.assertRaises(ValueError):
            process_conversation({"id": "x"}, "v_user", memory_file=self._mem)

    def test_validation_rejects_missing_speaker(self):
        bad = {"id": "x", "turns": [{"text": "hello"}]}
        with self.assertRaises(ValueError):
            process_conversation(bad, "v_user", memory_file=self._mem)

    def test_resistant_memory_risk_tendency(self):
        for _ in range(3):
            self._run(RESISTANT_IMPULSE, uid="risk_user")
        r   = self._run(RESISTANT_IMPULSE, uid="risk_user")
        mem = r["memory"]
        self.assertIn(mem["risk_tendency"], ("high", "medium"))

    def test_all_fixtures_produce_valid_metrics(self):
        for fixture in ALL_FIXTURES:
            r = self._run(fixture)
            m = r["metrics"]
            self.assertIn(m["compliance_level"],  ("high", "medium", "low"))
            self.assertIn(m["pushback_level"],     ("high", "medium", "low"))
            self.assertIn(m["decision_speed"],     ("fast", "medium", "slow", "undecided"))
            self.assertIn(m["spending_intent"],    ("impulse", "planned", "essential", "unclear"))
            self.assertIsInstance(m["impulse_flag"], bool)


class TestSummarizer(unittest.TestCase):
    """Tests for the deterministic fallback summary."""

    def setUp(self):
        self.ex   = BehaviorExtractor()
        self.calc = MetricsCalculator()

    def _summary(self, fixture: dict) -> str:
        raw     = self.ex.extract(fixture)
        metrics = self.calc.calculate(raw)
        return _fallback_summary(metrics, raw)

    def test_summary_is_string(self):
        s = self._summary(MOM_SUCCESS)
        self.assertIsInstance(s, str)

    def test_summary_nonempty(self):
        for fixture in ALL_FIXTURES:
            s = self._summary(fixture)
            self.assertGreater(len(s), 0, f"Empty summary for {fixture['id']}")

    def test_high_compliance_text(self):
        s = self._summary(MOM_SUCCESS)
        self.assertIn("receptive", s.lower())

    def test_no_decision_text(self):
        s = self._summary(DAD_FAIL)
        self.assertIn("decision", s.lower())

    def test_impulse_flag_in_summary(self):
        s = self._summary(RESISTANT_IMPULSE)
        # Should mention impulse somewhere
        self.assertTrue("impulse" in s.lower() or "resistant" in s.lower() or "resistance" in s.lower())


# ── Demo printer ─────────────────────────────────────────────────────────────

def _print_demo():
    ex    = BehaviorExtractor()
    calc  = MetricsCalculator()
    names = {
        "conv_001": "Compliant + Hesitant (mean)",
        "conv_002": "Resistant + Impulse (bad)",
        "conv_003": "Mom — Success",
        "conv_004": "Dad — Fail",
        "conv_005": "Finance Coach — Cheaper Option",
    }
    for fixture in ALL_FIXTURES:
        raw     = ex.extract(fixture)
        metrics = calc.calculate(raw)
        summary = _fallback_summary(metrics, raw)
        label   = names.get(fixture["id"], fixture["id"])
        print(f"\n== {label} ==")
        print("-- Raw Features --")
        for k, v in raw.items():
            print(f"  {k}: {v}")
        print("-- Metrics --")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("-- Summary --")
        print(f"  {summary}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--demo" in sys.argv:
        _print_demo()
    else:
        # Remove --demo from argv so unittest doesn't choke
        sys.argv = [a for a in sys.argv if a != "--demo"]
        unittest.main(verbosity=2)
