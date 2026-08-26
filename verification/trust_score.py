"""
verification/trust_score.py

Module 5 — Trust Score.

Instead of a single pass/fail verdict, this converts the herd and freshness
verification results into a weighted 0-100 trust score, then maps that score
to a final decision. This is the part your reference document calls "your
novelty" — a graded confidence measure rather than a binary check.

Weighting (100 points total):
    40 pts — Count match (declared vs. detected herd count)
    30 pts — Detection confidence (how sure the model was about its count)
    30 pts — Freshness result (label + confidence combined)

USAGE:
    from verification.trust_score import calculate_trust_score
    score, breakdown = calculate_trust_score(
        declared_count=12, detected_count=12, avg_detection_confidence=0.91,
        freshness_label="fresh", freshness_confidence=0.88
    )
"""


def _count_match_points(declared_count: int, detected_count: int, max_points: int = 40) -> float:
    """
    Full points if counts match exactly. Points fall off linearly as the
    difference grows, reaching 0 once the difference is 3 or more — beyond
    that, treat it as a full mismatch regardless of how much worse it gets.
    """
    difference = abs(declared_count - detected_count)
    if difference == 0:
        return max_points
    if difference >= 3:
        return 0.0
    # difference of 1 -> 2/3 of points, difference of 2 -> 1/3 of points
    return max_points * (1 - difference / 3)


def _detection_confidence_points(avg_confidence: float, max_points: int = 30) -> float:
    """Straightforward scaling — detection confidence is already 0-1."""
    return max_points * max(0.0, min(1.0, avg_confidence))


def _freshness_points(label: str, confidence: float, max_points: int = 30) -> float:
    """
    Full points only for a confident "fresh" result. Half_fresh gets partial
    credit even at high confidence (it's a real, usable-but-lesser quality
    tier, not a failure). Spoiled always scores 0, regardless of how
    confident the model is that it's spoiled — a confident "this is spoiled"
    should not accidentally score well.
    """
    label = label.lower()
    if label == "spoiled":
        return 0.0
    if label == "half_fresh":
        return max_points * 0.5 * max(0.0, min(1.0, confidence))
    if label == "fresh":
        return max_points * max(0.0, min(1.0, confidence))
    return 0.0  # unrecognized label — treat as no points rather than guessing


def calculate_trust_score(declared_count: int, detected_count: int, avg_detection_confidence: float,
                            freshness_label: str, freshness_confidence: float):
    """
    Returns (total_score: int, breakdown: dict) where breakdown shows each
    component's points earned out of its maximum — useful for the report
    page so a buyer/trustee can see WHY a score landed where it did, not
    just the final number.
    """
    count_points = _count_match_points(declared_count, detected_count)
    confidence_points = _detection_confidence_points(avg_detection_confidence)
    freshness_points = _freshness_points(freshness_label, freshness_confidence)

    total = round(count_points + confidence_points + freshness_points)

    breakdown = {
        "count_match": {"earned": round(count_points, 1), "max": 40},
        "detection_confidence": {"earned": round(confidence_points, 1), "max": 30},
        "freshness": {"earned": round(freshness_points, 1), "max": 30},
    }

    return total, breakdown


def score_to_decision(trust_score: int) -> str:
    """
    Maps the trust score to a final decision. Thresholds are a starting
    point — tune these once you see real score distributions during
    evaluation, and document why you chose them in your thesis.
    """
    if trust_score >= 85:
        return "Verified"
    if trust_score >= 60:
        return "Review Required"
    return "Rejected"
