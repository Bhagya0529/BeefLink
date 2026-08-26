"""
verification/verification_engine.py

Module 4 — Verification Engine. This is the core of your thesis
contribution: it takes the outputs of Module 2 (herd) and Module 3 (meat)
and combines them into one trust score and one final decision, using the
rule-based logic from your reference document PLUS the weighted trust score
from Module 5.

Both a simple rule-based decision AND the graded trust score are computed —
showing both in your evaluation lets you compare a binary rule-based
approach against the graded one, which is good material for your thesis
discussion (does grading actually produce better/fairer outcomes than a
hard rule cutoff?).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from verification.trust_score import calculate_trust_score, score_to_decision


def rule_based_decision(herd_result: dict, meat_result: dict) -> str:
    """
    The simple IF/AND rule from your reference document — kept as a baseline
    to compare the trust-score approach against.
    """
    count_matches = herd_result["difference"] == 0
    high_detection_confidence = herd_result["average_confidence"] > 0.85
    is_fresh = meat_result["label"] == "fresh"
    high_freshness_confidence = meat_result["confidence"] > 0.90

    if count_matches and high_detection_confidence and is_fresh and high_freshness_confidence:
        return "Verified"

    # Distinguish "close but not quite" from "clearly wrong" the same way
    # Module 2 already does for count mismatches.
    if herd_result["status"] == "rejected" or meat_result["label"] == "spoiled":
        return "Rejected"

    return "Review Required"


def run_verification(declared_count: int, herd_result: dict, meat_result: dict):
    """
    Combines herd + meat results into a full verification outcome.

    Returns a dict with:
        trust_score, trust_score_breakdown, trust_score_decision,
        rule_based_decision, final_decision
    """
    trust_score, breakdown = calculate_trust_score(
        declared_count=declared_count,
        detected_count=herd_result["detected_count"],
        avg_detection_confidence=herd_result["average_confidence"],
        freshness_label=meat_result["label"],
        freshness_confidence=meat_result["confidence"],
    )
    trust_decision = score_to_decision(trust_score)
    rule_decision = rule_based_decision(herd_result, meat_result)

    return {
        "trust_score": trust_score,
        "trust_score_breakdown": breakdown,
        "trust_score_decision": trust_decision,
        "rule_based_decision": rule_decision,
        # The trust score is the primary decision — it's the graded,
        # explainable approach this thesis argues for. Rule-based is kept
        # alongside it for comparison, not as the final word.
        "final_decision": trust_decision,
    }
