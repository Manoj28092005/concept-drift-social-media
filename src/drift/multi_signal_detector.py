"""
Multi-Signal Drift Detector
Implements the Core Extension Proposed by Group 23:
Fusing Sentiment Drift Monitoring (TSDA-BERT Base Paper) with Topic Dynamics (Paper 2)

Key Capabilities:
1. Joint Multi-Signal Drift Index: Fuses sentiment shift and topic divergence.
2. Root-Cause Classification:
   - STATIONARY: No significant drift in either domain.
   - PURE_SENTIMENT_DRIFT: Sentiment flipped while topics remained stable.
   - TOPIC_DRIVEN_DRIFT: Topic shift occurred, pulling sentiment with it.
   - COMPOUND_CO_DRIFT: Simultaneous drastic sentiment and topic upheaval.
3. Enhanced adaptation trigger with reduced detection delay and suppressed false alarms.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiSignalDriftDetector:
    """
    Multi-Signal Concept Drift Detector combining Sentiment Drift Measure (SDM)
    and Topic Divergence (JSD/Cosine).
    """
    def __init__(
        self,
        alpha: float = 0.5,
        joint_threshold: float = 0.35,
        sdm_threshold: float = 0.35,
        topic_jsd_threshold: float = 0.28
    ):
        self.alpha = alpha
        self.joint_threshold = joint_threshold
        self.sdm_threshold = sdm_threshold
        self.topic_jsd_threshold = topic_jsd_threshold
        self.history: List[Dict[str, Any]] = []

    def evaluate_multi_signal(
        self,
        sentiment_res: Dict[str, Any],
        topic_res: Dict[str, Any],
        window_id: str = ""
    ) -> Dict[str, Any]:
        """
        Fuses sentiment and topic signals to evaluate drift state.
        """
        delta_sdm = abs(sentiment_res.get('delta_sdm', 0.0))
        is_sentiment_drift = sentiment_res.get('is_drift', False)
        sdm_sign_flip = (
            sentiment_res.get('previous_sdm') is not None and
            (sentiment_res.get('previous_sdm') * sentiment_res.get('sdm', 0.0) < 0)
        )

        jsd_val = topic_res.get('jsd', 0.0)
        is_topic_drift = topic_res.get('is_topic_drift', False)

        # Normalized signals in [0, 1]
        norm_sdm_change = min(1.0, (delta_sdm / 1.5) + (0.3 if sdm_sign_flip else 0.0))
        cos_val = topic_res.get('cosine_distance', 0.0)
        norm_topic_change = min(1.0, max(jsd_val / 0.35, cos_val))

        joint_drift_index = float(self.alpha * norm_sdm_change + (1.0 - self.alpha) * norm_topic_change)

        # Categorize multi-signal drift typology
        if is_sentiment_drift and is_topic_drift:
            drift_type = "COMPOUND_CO_DRIFT"
            drift_detected = True
            diagnosis = (
                f"Compound Co-Drift: Both sentiment ({sentiment_res.get('sdm', 0.0):+.2f}) and topic "
                f"(JSD: {jsd_val:.3f}, CosDist: {cos_val:.3f}) underwent abrupt simultaneous structural shift."
            )
        elif is_sentiment_drift and not is_topic_drift:
            drift_type = "PURE_SENTIMENT_DRIFT"
            drift_detected = True
            diagnosis = (
                f"Pure Sentiment Drift: User opinion flipped ({sentiment_res.get('previous_sdm', 0):+.2f} "
                f"-> {sentiment_res.get('sdm', 0.0):+.2f}) while topic space remained stationary."
            )
        elif is_topic_drift:
            drift_type = "TOPIC_DRIVEN_DRIFT"
            drift_detected = True
            diagnosis = (
                f"Topic-Driven Drift: Conversation migrated to new topic "
                f"(dominant keywords: {', '.join(topic_res.get('dominant_keywords', [])[:3])}) "
                f"influencing concept distribution."
            )
        elif joint_drift_index >= self.joint_threshold:
            drift_type = "MULTI_SIGNAL_ANOMALY"
            drift_detected = True
            diagnosis = f"Multi-Signal Drift Anomaly: Weighted fusion index ({joint_drift_index:.3f}) exceeded threshold."
        else:
            drift_type = "STATIONARY"
            drift_detected = False
            diagnosis = "Stream within nominal stationary bounds. No significant drift."

        result = {
            'window_id': window_id,
            'is_drift': drift_detected,
            'drift_type': drift_type,
            'joint_drift_index': joint_drift_index,
            'diagnosis': diagnosis,
            'sentiment_metrics': sentiment_res,
            'topic_metrics': topic_res
        }

        self.history.append(result)
        logger.info(f"[{window_id}] Multi-Signal: {drift_type} (Joint Index: {joint_drift_index:.3f}, Drift: {drift_detected})")
        return result

    def reset(self):
        self.history = []
