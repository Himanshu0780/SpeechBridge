"""
utils/metrics.py
-----------------
Evaluation metrics for ASR and translation quality.
Implements Word Error Rate (WER) and BLEU score.
"""

import re
import math
import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison in metrics.
    Lowercases, removes punctuation (except apostrophes), strips extra whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> Dict:
    """
    Compute Word Error Rate (WER) between reference and hypothesis transcripts.

    WER = (Substitutions + Deletions + Insertions) / len(reference)

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        normalize: Whether to normalize text before comparison

    Returns:
        Dict with wer, substitutions, deletions, insertions, ref_words, hyp_words
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return {
            "wer": 0.0 if len(hyp_words) == 0 else 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "ref_words": 0,
            "hyp_words": len(hyp_words),
        }

    # Dynamic programming for edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1], # substitution
                )

    total_edits = dp[n][m]
    wer = total_edits / n

    # Approximate breakdown (exact needs backtracking)
    return {
        "wer": round(min(wer, 1.0), 4),
        "wer_percent": round(min(wer * 100, 100), 2),
        "total_edits": total_edits,
        "ref_words": n,
        "hyp_words": m,
        "accuracy_percent": round(max(0, (1 - wer) * 100), 2),
    }


def compute_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    normalize: bool = True,
) -> Dict:
    """
    Compute BLEU score for translation quality evaluation.

    Args:
        reference: Ground truth translation
        hypothesis: Model output translation
        max_n: Maximum n-gram order (typically 4)
        normalize: Whether to normalize text

    Returns:
        Dict with bleu, precision scores, brevity_penalty
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0:
        return {"bleu": 0.0, "bleu_percent": 0.0, "precision_scores": []}

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    # Compute n-gram precisions
    precision_scores = []

    for n in range(1, max_n + 1):
        # Get n-grams from reference and hypothesis
        ref_ngrams = Counter(_get_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_get_ngrams(hyp_tokens, n))

        # Clipped count
        clipped_count = sum(
            min(count, ref_ngrams[gram])
            for gram, count in hyp_ngrams.items()
        )
        total_count = max(len(hyp_tokens) - n + 1, 0)

        if total_count == 0:
            precision_scores.append(0.0)
        else:
            precision_scores.append(clipped_count / total_count)

    # BLEU = BP * exp(mean of log precisions)
    if all(p == 0 for p in precision_scores):
        bleu = 0.0
    else:
        log_avg = sum(
            math.log(p) if p > 0 else float("-inf")
            for p in precision_scores
        ) / max_n
        bleu = bp * math.exp(log_avg)

    return {
        "bleu": round(bleu, 4),
        "bleu_percent": round(bleu * 100, 2),
        "brevity_penalty": round(bp, 4),
        "precision_scores": [round(p, 4) for p in precision_scores],
        "ref_length": ref_len,
        "hyp_length": hyp_len,
    }


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def try_sacrebleu(reference: str, hypothesis: str) -> Optional[float]:
    """
    Use sacrebleu library for official BLEU score (more accurate).
    Falls back to our implementation if not available.
    """
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        result = bleu.sentence_score(hypothesis, [reference])
        return result.score
    except ImportError:
        logger.debug("sacrebleu not available, using built-in BLEU")
        return None
    except Exception as e:
        logger.debug(f"sacrebleu error: {e}")
        return None


def evaluate_translation(
    reference_text: str,
    hypothesis_text: str,
    reference_transcript: Optional[str] = None,
    hypothesis_transcript: Optional[str] = None,
) -> Dict:
    """
    Full evaluation: BLEU for translation + optional WER for ASR.

    Args:
        reference_text: Gold-standard translation
        hypothesis_text: Model translation output
        reference_transcript: Optional reference ASR transcript
        hypothesis_transcript: Optional model ASR output

    Returns:
        Dict with all metrics
    """
    results = {}

    # BLEU score
    sacrebleu_score = try_sacrebleu(reference_text, hypothesis_text)
    bleu_results = compute_bleu(reference_text, hypothesis_text)

    results["bleu"] = {
        "score": sacrebleu_score if sacrebleu_score else bleu_results["bleu_percent"],
        "method": "sacrebleu" if sacrebleu_score else "custom",
        **bleu_results,
    }

    # WER (if transcripts provided)
    if reference_transcript and hypothesis_transcript:
        results["wer"] = compute_wer(reference_transcript, hypothesis_transcript)

    return results
