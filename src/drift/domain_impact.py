"""
Domain Impact Score (DIS) & Relevant Tweet Selector
Implements Section 3.5.4 of Susi & Shanthi (2023):
"Retraining of BERT with Recent Tweets Based on Sentiment Drift"

Formula:
  domain_impact_score = |W ∩ D| / |W|
where:
  W = set of words in a tweet {w_1, w_2, ..., w_n}
  D = set of domain-specific impact keywords {d_1, d_2, ..., d_m}

Tweets with DIS >= threshold (e.g., 0.5) are isolated to adapt the sentiment model
without corrupting it with irrelevant or noisy streaming text.
"""

from typing import List, Dict, Any, Set, Tuple
from ..preprocessing import preprocessor
import logging

logger = logging.getLogger(__name__)


class DomainImpactScorer:
    """
    Computes Domain Impact Score (DIS) and filters high-relevance domain tweets.
    """
    def __init__(self, domain_lexicon: Set[str], threshold: float = 0.35):
        self.domain_lexicon = {w.lower() for w in domain_lexicon}
        self.threshold = threshold

    def compute_dis(self, tweet_text: str) -> Tuple[float, Set[str], int]:
        """
        Calculates DIS for a single tweet.
        Returns:
            dis_score: float in [0.0, 1.0]
            matched_words: set of intersection words (W ∩ D)
            total_words: count of words |W|
        """
        words = preprocessor.extract_word_set(tweet_text)
        if not words:
            return 0.0, set(), 0

        intersection = words.intersection(self.domain_lexicon)
        score = len(intersection) / len(words)
        return float(score), intersection, len(words)

    def filter_relevant_tweets(
        self,
        tweets: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filters tweets passing the domain impact threshold (DIS >= threshold).
        Returns:
            selected_tweets: list of tweets qualifying for model adaptation
            stats: summary metrics of selection
        """
        selected = []
        all_scores = []

        for t in tweets:
            text = t.get('text', '')
            score, matched, total_w = self.compute_dis(text)
            all_scores.append(score)

            # Store score on tweet object for auditability
            t_copy = dict(t)
            t_copy['domain_impact_score'] = score
            t_copy['matched_domain_words'] = list(matched)

            if score >= self.threshold:
                selected.append(t_copy)

        avg_dis = float(sum(all_scores) / max(1, len(all_scores)))
        selection_rate = len(selected) / max(1, len(tweets))

        stats = {
            'total_evaluated': len(tweets),
            'total_selected': len(selected),
            'selection_rate': selection_rate,
            'average_dis': avg_dis,
            'threshold': self.threshold
        }

        logger.info(f"DIS Selection: {len(selected)}/{len(tweets)} tweets selected (rate: {selection_rate:.1%}, avg DIS: {avg_dis:.3f})")
        return selected, stats
