"""
Confidence scoring — combines Signal 1 (LLM) and Signal 2 (stylometric)
into a single score in [0.0, 1.0].

Formula:
    confidence = 0.7 × llm_score + 0.3 × stylo_score

Signal 1 carries more weight (0.7) because it captures semantic meaning
holistically. Signal 2 (0.3) adds a structural cross-check.

Signal disagreement is surfaced via the `signal_gap` field — a large gap
between the two scores means the signals disagree, which the transparency
label can communicate as additional uncertainty.

Label thresholds:
    0.00 - 0.39  →  "likely_human"
    0.40 - 0.69  →  "uncertain"
    0.70 - 1.00  →  "likely_ai"

These are initial estimates to be validated during M4 testing and
recalibrated if all three labels aren't reachable.
"""

# Weights must sum to 1.0
LLM_WEIGHT = 0.7
STYLO_WEIGHT = 0.3

# Label thresholds — conservative to minimize false positives on human work
THRESHOLD_HUMAN = 0.40   # below this → likely_human
THRESHOLD_AI = 0.70      # above this → likely_ai
                          # between   → uncertain


def compute_confidence(llm_score: float, stylo_score: float) -> dict:
    """
    Combines both signal scores into a single confidence score and
    maps it to an attribution label.

    Args:
        llm_score:   Signal 1 output, float in [0.0, 1.0]
        stylo_score: Signal 2 output, float in [0.0, 1.0]

    Returns a dict:
        {
            "confidence":  float,  # combined score [0.0, 1.0]
            "attribution": str,    # "likely_ai" | "uncertain" | "likely_human"
            "signal_gap":  float,  # abs difference between signals
        }
    """
    confidence = LLM_WEIGHT * llm_score + STYLO_WEIGHT * stylo_score
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    if confidence >= THRESHOLD_AI:
        attribution = "likely_ai"
    elif confidence >= THRESHOLD_HUMAN:
        attribution = "uncertain"
    else:
        attribution = "likely_human"

    signal_gap = round(abs(llm_score - stylo_score), 4)

    return {
        "confidence": confidence,
        "attribution": attribution,
        "signal_gap": signal_gap,
    }


# ── manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("both strongly AI",    0.90, 0.80),
        ("both strongly human", 0.10, 0.15),
        ("LLM says AI, stylo uncertain", 0.80, 0.40),
        ("LLM uncertain, stylo says human", 0.45, 0.10),
        ("signals disagree strongly", 0.85, 0.15),
        ("both middle",         0.50, 0.50),
    ]

    print(f"{'Case':<35} {'LLM':>5}  {'Stylo':>5}  {'Conf':>5}  {'Gap':>5}  Attribution")
    print("-" * 80)
    for label, llm, stylo in test_cases:
        r = compute_confidence(llm, stylo)
        print(
            f"{label:<35} {llm:>5.2f}  {stylo:>5.2f}  "
            f"{r['confidence']:>5.3f}  {r['signal_gap']:>5.3f}  {r['attribution']}"
        )