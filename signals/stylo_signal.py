"""
Signal 2: Stylometric heuristics — pure Python, no inference.

Measures two structural properties of text and combines them into a
score in [0.0, 1.0]:
  0.0 = structurally matches expected human writing
  1.0 = structurally matches expected AI-generated writing

The two metrics are:

  1. Sentence length variance (stdev of word counts per sentence)
       AI text tends toward uniform sentence lengths → low stdev → high score
       Human text varies more → high stdev → low score
       Normalized against STDEV_CEILING (8 words), then inverted.

  2. Average word length (mean character count per word)
       AI text favors formal, polysyllabic vocabulary → longer avg word length
       Human casual writing uses shorter, simpler words
       Normalized against WORD_LEN_CEILING (7 chars), then kept as-is
       (longer words → higher score, no inversion needed).
       Length-independent: computed per word, not per text.

       Why not TTR: TTR is length-sensitive. At 40–60 words, all texts
       use nearly every word once, making TTR indistinguishable across
       samples of similar length. Windowed TTR at 50 words didn't help
       because most samples are already under 60 words. Average word
       length captures the formal/casual vocabulary dimension TTR was
       meant to proxy, without the length dependency.

Combined: stylo_score = 0.5 x stdev_score + 0.5 x wordlen_score

Test independently before wiring into app.py:
    python signals/stylo_signal.py
"""

import re
import math

STDEV_CEILING = 8.0     # stdev above this → maximally human-like (score → 0.0)
WORD_LEN_CEILING = 7.0  # avg word length above this → maximally AI-like (score → 1.0)
                         # English avg is ~4.5 chars; formal/AI text ~5.5–6.5


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _word_count(sentence: str) -> int:
    return len(sentence.split())


def _stdev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _avg_word_length(words: list[str]) -> float:
    """Mean character count per word, stripping punctuation."""
    cleaned = [re.sub(r'[^a-zA-Z]', '', w) for w in words]
    cleaned = [w for w in cleaned if w]  # drop empty after stripping
    if not cleaned:
        return 0.0
    return sum(len(w) for w in cleaned) / len(cleaned)


def get_stylo_score(text: str) -> dict:
    """
    Returns a dict with the combined stylo score and both raw metrics:
        {
            "stylo_score":    float,  # combined [0.0, 1.0]
            "stdev_score":    float,  # sentence length uniformity component
            "wordlen_score":  float,  # vocabulary formality component
            "raw_stdev":      float,  # raw sentence length stdev (words)
            "raw_wordlen":    float,  # raw average word length (chars)
        }

    Returns stylo_score=0.5 (uncertain) for very short texts where
    the metrics are unreliable (fewer than 2 sentences or 10 words).
    """
    sentences = _split_sentences(text)
    words = text.lower().split()

    # Guard: too short to measure reliably
    if len(sentences) < 2 or len(words) < 10:
        return {
            "stylo_score": 0.5,
            "stdev_score": 0.5,
            "wordlen_score": 0.5,
            "raw_stdev": 0.0,
            "raw_wordlen": 0.0,
        }

    # ── Metric 1: sentence length variance ───────────────────────────────────
    lengths = [_word_count(s) for s in sentences]
    raw_stdev = _stdev(lengths)
    normalized_stdev = min(raw_stdev / STDEV_CEILING, 1.0)
    # Invert: low variance (AI-like) → stdev_score near 1.0
    stdev_score = 1.0 - normalized_stdev

    # ── Metric 2: average word length ─────────────────────────────────────────
    raw_wordlen = _avg_word_length(words)
    # No inversion: longer words → more formal/AI-like → higher score
    wordlen_score = min(raw_wordlen / WORD_LEN_CEILING, 1.0)

    # ── Combine ───────────────────────────────────────────────────────────────
    stylo_score = 0.5 * stdev_score + 0.5 * wordlen_score

    return {
        "stylo_score": round(stylo_score, 4),
        "stdev_score": round(stdev_score, 4),
        "wordlen_score": round(wordlen_score, 4),
        "raw_stdev": round(raw_stdev, 4),
        "raw_wordlen": round(raw_wordlen, 4),
    }


# ── manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        (
            "clearly AI",
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment.",
        ),
        (
            "clearly human",
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better. probably won't go back "
            "unless someone drags me there",
        ),
        (
            "Gemini generated poem",
            """A cold ceramic mug against my palm,
The radiator hums its heavy tune,
And for a second, everything is calm,
Beneath the pale, indifferent afternoon.

I think of lines I meant to write last week,
Of keys I lost, and words I shouldn't say,
The quiet, stubborn ways we try to speak
Across the gaps we widen day by day.""",
        ),
        (
            "human - William Faulkner poem",
            """The race's splendor lifts her lip, exposes
Amid her scarlet smile her little teeth;
The years are sand the wind plays with; beneath
The prisoned music of her deathless roses.

Within frostbitten rock she's fixed and glassed;
Now man may look upon her without fear.
But her contemptuous eyes back through him stare
And shear his fatuous sheep when he has passed.""",
        ),
        (
            "AI with human edits",
            "Additionally, Athens' religious practices and devotion are symbolized "
            "by the Parthenon. Athena's temple reflected ideals relevant to a variety "
            "of Greeks, despite being their patron goddess. Some of these ideals "
            "included wisdom, protection, and civic strength. Despite being a subject, "
            "I still felt a sense of shared reverence every time I saw the Parthenon.",
        ),
        (
            "human - personal notes",
            "- Found this book through @lifeonbooks instagram\n"
            "- Plot: The crazy events of a well-known family's anniversary party, "
            "with the father a prominent magistrate in Bogota, Colombia\n"
            "- Overall, a book that kept you on your toes",
        ),
        (
            "Clearly AI-generated (should score high)",
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment.",
        ),
        (
            "Clearly human-written (should score low)",
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better. probably won't go back "
            "unless someone drags me there",
        ),
        (
            "Borderline: formal human writing",
            "The relationship between monetary policy and asset price inflation "
            "has been extensively studied in the literature. Central banks face "
            "a fundamental tension between their mandate for price stability and "
            "the unintended consequences of prolonged low interest rates on "
            "equity and real estate valuations.",
        ),
        (
            "Borderline: lightly edited AI output",
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type.",
        ),
    ]

    print(f"{'Label':<45} {'Stylo':>6}  {'StDev':>6}  {'WdLen':>6}  {'raw_sd':>7}  {'raw_wl':>7}")
    print("-" * 92)
    for label, text in samples:
        r = get_stylo_score(text)
        verdict = "AI  " if r["stylo_score"] >= 0.70 else ("????" if r["stylo_score"] >= 0.40 else "HUM ")
        print(
            f"{label:<45} {r['stylo_score']:>6.3f}  "
            f"{r['stdev_score']:>6.3f}  {r['wordlen_score']:>6.3f}  "
            f"{r['raw_stdev']:>7.2f}  {r['raw_wordlen']:>7.3f}  {verdict}"
        )