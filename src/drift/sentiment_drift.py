"""
Sentiment Drift Measure (SDM)
Implements Section 3.5.3 of Susi & Shanthi (2023):
"Sentiment Classification and Drift Detection"

Formulas:
- positive_score_t = (number of positive tweets) / (total number of tweets)
- negative_score_t = (number of negative tweets) / (total number of tweets)
- SDM_t = positive_score_t - negative_score_t

Drift condition:
- Change in score from positive to negative and vice-versa between consecutive windows,
  or magnitude difference |SDM_t - SDM_{t-1}| exceeding threshold.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentDriftDetector:
    """
    Baseline TSDA-BERT Sentiment Drift Detector.
    Monitors positive/negative scores and SDM across streaming windows.
    """
    def __init__(self, magnitude_threshold: float = 0.35):
        self.magnitude_threshold = magnitude_threshold
        self.previous_sdm: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    def compute_window_sentiment(self, predictions: List[int]) -> Dict[str, float]:
        """
        Computes positive score, negative score, and SDM for a window.
        predictions: list of 0 (negative) or 1 (positive)
        """
        total = len(predictions)
        if total == 0:
            return {'pos_score': 0.0, 'neg_score': 0.0, 'sdm': 0.0, 'total': 0}

        num_pos = sum(1 for p in predictions if p == 1)
        num_neg = sum(1 for p in predictions if p == 0)

        pos_score = num_pos / total
        neg_score = num_neg / total
        sdm = pos_score - neg_score

        return {
            'pos_score': float(pos_score),
            'neg_score': float(neg_score),
            'sdm': float(sdm),
            'num_pos': num_pos,
            'num_neg': num_neg,
            'total': total
        }

    def detect_drift(self, predictions: List[int], window_id: str = "") -> Dict[str, Any]:
        """
        Calculates SDM for the current window and checks for drift against previous window.
        """
        metrics = self.compute_window_sentiment(predictions)
        current_sdm = metrics['sdm']

        is_drift = False
        drift_reason = "No drift"
        delta_sdm = 0.0

        if self.previous_sdm is not None:
            delta_sdm = current_sdm - self.previous_sdm
            sign_flip = (self.previous_sdm * current_sdm < 0)
            magnitude_shift = abs(delta_sdm) >= self.magnitude_threshold

            if sign_flip or magnitude_shift:
                is_drift = True
                drift_reason = (
                    f"Sentiment polarity flipped ({self.previous_sdm:+.2f} -> {current_sdm:+.2f})"
                    if sign_flip else
                    f"Sentiment magnitude shifted by {abs(delta_sdm):.2f} (threshold: {self.magnitude_threshold})"
                )

        result = {
            'window_id': window_id,
            'pos_score': metrics['pos_score'],
            'neg_score': metrics['neg_score'],
            'sdm': current_sdm,
            'previous_sdm': self.previous_sdm,
            'delta_sdm': delta_sdm,
            'is_drift': is_drift,
            'drift_reason': drift_reason,
            'total_tweets': metrics['total']
        }

        self.previous_sdm = current_sdm
        self.history.append(result)
        return result

    def reset(self):
        self.previous_sdm = None
        self.history = []
