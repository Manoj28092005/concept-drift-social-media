"""
Adaptive Model Updater & Sliding Memory Buffer
Implements Section 3.5.4 of Susi & Shanthi (2023):
"Retraining of BERT with Recent Tweets Based on Sentiment Drift"

Supports:
1. Standard FIFO sliding window memory update under nominal (no-drift) conditions.
2. Drift-triggered selective adaptation using high Domain Impact Score (DIS) tweets.
3. Multi-signal topic-aware sampling to prevent catastrophic forgetting.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
from ..models.bert_sentiment import TSDABERTClassifier
from ..drift.domain_impact import DomainImpactScorer

logger = logging.getLogger(__name__)


class AdaptiveModelUpdater:
    """
    Manages the adaptive training dataset buffer and online retraining of TSDA-BERT.
    """
    def __init__(
        self,
        classifier: TSDABERTClassifier,
        dis_scorer: DomainImpactScorer,
        max_buffer_size: int = 300,
        adaptation_epochs: int = 3,
        adaptation_lr: float = 5e-5
    ):
        self.classifier = classifier
        self.dis_scorer = dis_scorer
        self.max_buffer_size = max_buffer_size
        self.adaptation_epochs = adaptation_epochs
        self.adaptation_lr = adaptation_lr

        # Persistent training buffer (texts and labels)
        self.training_texts: List[str] = []
        self.training_labels: List[int] = []
        self.adaptation_event_log: List[Dict[str, Any]] = []

    def initialize_buffer(self, initial_texts: List[str], initial_labels: List[int]):
        """Seeds the buffer with historical tweets (T0)."""
        self.training_texts = list(initial_texts[:self.max_buffer_size])
        self.training_labels = list(initial_labels[:self.max_buffer_size])
        logger.info(f"Initialized training memory buffer with {len(self.training_texts)} instances.")

    def update_on_window(
        self,
        window_tweets: List[Dict[str, Any]],
        is_drift: bool,
        use_dis_filtering: bool = True,
        window_id: str = ""
    ) -> Dict[str, Any]:
        """
        Executes window update procedure as specified in Section 3.5.4.
        If is_drift is False: standard FIFO update.
        If is_drift is True: select high DIS tweets, evict oldest instances, retrain classifier.
        """
        raw_texts = [t.get('text', '') for t in window_tweets]
        labels = [t.get('sentiment', 0) for t in window_tweets]

        adaptation_occurred = False
        selected_count = 0
        evicted_count = 0
        dis_stats = {}

        if is_drift:
            # Drift condition
            if use_dis_filtering:
                selected_tweets, dis_stats = self.dis_scorer.filter_relevant_tweets(window_tweets)
                cand_texts = [t.get('text', '') for t in selected_tweets]
                cand_labels = [t.get('sentiment', 0) for t in selected_tweets]
            else:
                cand_texts = raw_texts
                cand_labels = labels
                dis_stats = {'note': 'DIS filtering disabled for baseline comparison'}

            if len(cand_texts) > 0:
                selected_count = len(cand_texts)
                # Append candidate tweets to buffer
                self.training_texts.extend(cand_texts)
                self.training_labels.extend(cand_labels)

                # Evict oldest tweets to maintain capacity bounds
                if len(self.training_texts) > self.max_buffer_size:
                    overflow = len(self.training_texts) - self.max_buffer_size
                    evicted_count = overflow
                    self.training_texts = self.training_texts[overflow:]
                    self.training_labels = self.training_labels[overflow:]

                # Execute online adaptation / fine-tuning
                self.classifier.adapt(
                    adaptation_texts=cand_texts,
                    adaptation_labels=cand_labels,
                    epochs=self.adaptation_epochs,
                    lr=self.adaptation_lr
                )
                adaptation_occurred = True
        else:
            # Nominal (no-drift) FIFO update
            new_count = min(len(raw_texts), 20)  # Moderate FIFO ingress
            if new_count > 0:
                self.training_texts.extend(raw_texts[:new_count])
                self.training_labels.extend(labels[:new_count])
                if len(self.training_texts) > self.max_buffer_size:
                    overflow = len(self.training_texts) - self.max_buffer_size
                    self.training_texts = self.training_texts[overflow:]
                    self.training_labels = self.training_labels[overflow:]

        event_record = {
            'window_id': window_id,
            'is_drift': is_drift,
            'adaptation_occurred': adaptation_occurred,
            'use_dis_filtering': use_dis_filtering,
            'selected_tweets_count': selected_count,
            'evicted_tweets_count': evicted_count,
            'current_buffer_size': len(self.training_texts),
            'dis_stats': dis_stats
        }

        self.adaptation_event_log.append(event_record)
        return event_record
