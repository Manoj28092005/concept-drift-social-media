"""
Topic Modeling & Semantic Representation Module
Implements the core mechanism from Paper 2:
"Concept Drift Influenced by Topic Change in Data Streams from Social Media" (Sarnovský & Babič, IEEE SAMI 2025).

Extracts dynamic topic distributions and semantic centroid embeddings for each streaming window,
allowing the detector to track how conversational topics shift and trigger concept drift.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


class StreamTopicModel:
    """
    Online/Window-based Topic Model that extracts topic distributions and keywords
    across consecutive streaming windows.
    """
    def __init__(self, n_topics: int = 3, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=2)
        self.cluster_model = MiniBatchKMeans(n_clusters=n_topics, random_state=random_state, batch_size=64)
        self.is_fitted = False
        self.vocab = []

    def fit(self, texts: List[str]):
        """Fits initial topic space on baseline corpus."""
        if len(texts) < self.n_topics:
            return

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.cluster_model.fit(tfidf_matrix)
        self.is_fitted = True
        self.vocab = np.array(self.vectorizer.get_feature_names_out())
        logger.info(f"StreamTopicModel fitted with {self.n_topics} topic clusters.")

    def transform_window(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
        """
        Transforms a streaming window of tweets.
        Returns:
            topic_distribution: 1D probability distribution array of length n_topics (sums to 1.0)
            centroid_embedding: 1D mean TF-IDF vector representing the window centroid
            top_keywords: dictionary mapping topic_id -> list of representative keywords
        """
        if not self.is_fitted:
            self.fit(texts)

        if len(texts) == 0:
            uniform = np.ones(self.n_topics) / self.n_topics
            return uniform, np.zeros(500), {}

        tfidf_matrix = self.vectorizer.transform(texts)
        dense_matrix = tfidf_matrix.toarray()

        # Window centroid
        centroid = np.mean(dense_matrix, axis=0)

        # Topic assignment per tweet
        cluster_assignments = self.cluster_model.predict(tfidf_matrix)
        counts = np.bincount(cluster_assignments, minlength=self.n_topics)
        topic_distribution = counts / max(1, len(texts))

        # Extract top keywords per topic cluster
        top_keywords = {}
        order_centroids = self.cluster_model.cluster_centers_
        for i in range(self.n_topics):
            top_word_indices = order_centroids[i].argsort()[::-1][:5]
            if len(self.vocab) > 0:
                words = [self.vocab[idx] for idx in top_word_indices if idx < len(self.vocab)]
            else:
                words = []
            top_keywords[i] = words

        return topic_distribution, centroid, top_keywords
