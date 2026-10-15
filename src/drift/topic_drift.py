"""
Topic Drift Detector
Implements Paper 2:
"Concept Drift Influenced by Topic Change in Data Streams from Social Media" (Sarnovský & Babič, IEEE SAMI 2025).

Measures:
1. Jensen-Shannon Divergence (JSD) between consecutive window topic distributions P_t and P_{t-1}.
2. Cosine distance between semantic centroid embeddings c_t and c_{t-1}.
Flags topic drift when statistical divergence exceeds threshold.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TopicDriftDetector:
    """
    Monitors and quantifies topic shifts in streaming social media text.
    """
    def __init__(self, jsd_threshold: float = 0.28, cosine_threshold: float = 0.25):
        self.jsd_threshold = jsd_threshold
        self.cosine_threshold = cosine_threshold
        self.previous_topic_dist: Optional[np.ndarray] = None
        self.previous_centroid: Optional[np.ndarray] = None
        self.history: List[Dict[str, Any]] = []

    def detect_topic_drift(
        self,
        current_topic_dist: np.ndarray,
        current_centroid: np.ndarray,
        window_id: str = "",
        keywords: Optional[Dict[int, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates topic divergence between consecutive windows.
        """
        # Ensure proper normalization
        p = np.array(current_topic_dist, dtype=np.float64)
        p = p / max(1e-9, np.sum(p))

        is_drift = False
        jsd_val = 0.0
        cos_dist = 0.0
        reason = "Baseline / No previous window"

        if self.previous_topic_dist is not None:
            q = np.array(self.previous_topic_dist, dtype=np.float64)
            q = q / max(1e-9, np.sum(q))

            # Jensen-Shannon Divergence
            jsd_val = float(jensenshannon(p, q, base=2))

            # Cosine distance between semantic centroids
            if self.previous_centroid is not None and np.linalg.norm(current_centroid) > 0 and np.linalg.norm(self.previous_centroid) > 0:
                cos_dist = float(cosine(current_centroid, self.previous_centroid))

            if jsd_val >= self.jsd_threshold or cos_dist >= self.cosine_threshold:
                is_drift = True
                reason = f"Topic distribution shift (JSD: {jsd_val:.3f} >= {self.jsd_threshold}, CosDist: {cos_dist:.3f})"

        # Identify dominant and emerging topics
        dominant_topic_id = int(np.argmax(p))
        top_words = keywords.get(dominant_topic_id, []) if keywords else []

        result = {
            'window_id': window_id,
            'topic_distribution': p.tolist(),
            'dominant_topic_id': dominant_topic_id,
            'dominant_keywords': top_words,
            'jsd': jsd_val,
            'cosine_distance': cos_dist,
            'is_topic_drift': is_drift,
            'reason': reason
        }

        self.previous_topic_dist = p
        self.previous_centroid = current_centroid
        self.history.append(result)
        return result

    def reset(self):
        self.previous_topic_dist = None
        self.previous_centroid = None
        self.history = []
