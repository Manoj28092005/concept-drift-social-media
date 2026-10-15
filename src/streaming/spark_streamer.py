"""
Sliding Window & Spark Streaming Processor
Implements Section 3.5.2 of Susi & Shanthi (2023):
"Implementation of Sliding Window with Spark for Parallel Sentiment Analysis"

Maintains a sliding window buffer W_t, evicting the oldest tweets when new streaming tweets arrive,
and dispatches micro-batches for parallel sentiment scoring and drift analysis.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """
    Sliding window buffer that maintains a fixed maximum capacity of tweets (FIFO),
    calculates positive/negative distributions, and emits windows for classification and drift monitoring.
    """
    def __init__(self, capacity: int = 200, window_id_prefix: str = "w"):
        self.capacity = capacity
        self.window_id_prefix = window_id_prefix
        self.buffer: List[Dict[str, Any]] = []
        self.current_window_idx = 0

    def add_tweets(self, new_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adds incoming streaming tweets to window.
        Evicts oldest tweets to maintain capacity (Section 3.5.2).
        """
        self.buffer.extend(new_tweets)
        evicted_count = 0
        if len(self.buffer) > self.capacity:
            evicted_count = len(self.buffer) - self.capacity
            self.buffer = self.buffer[-self.capacity:]

        window_id = f"{self.window_id_prefix}{self.current_window_idx}"
        self.current_window_idx += 1

        return {
            'window_id': window_id,
            'window_index': self.current_window_idx - 1,
            'current_size': len(self.buffer),
            'incoming_size': len(new_tweets),
            'evicted_size': evicted_count,
            'tweets': list(self.buffer)
        }

    def get_current_window(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def reset(self):
        self.buffer = []
        self.current_window_idx = 0


class SparkStreamingSimulator:
    """
    Spark-compatible streaming wrapper.
    Converts incoming sliding window micro-batches into PySpark or Pandas DataFrames
    for distributed transformation and inference.
    """
    def __init__(self, use_pyspark: bool = False):
        self.use_pyspark = use_pyspark
        self.spark = None

        if self.use_pyspark:
            try:
                from pyspark.sql import SparkSession
                self.spark = SparkSession.builder \
                    .appName("TSDA-BERT-Streaming") \
                    .master("local[2]") \
                    .config("spark.driver.memory", "2g") \
                    .getOrCreate()
                logger.info("Initialized local PySpark Session for streaming execution.")
            except Exception as e:
                logger.warning(f"Could not initialize PySpark: {e}. Defaulting to high-speed vectorized engine.")
                self.use_pyspark = False

    def to_dataframe(self, tweet_records: List[Dict[str, Any]]):
        """Converts tweet window records to appropriate dataframe format."""
        pdf = pd.DataFrame(tweet_records)
        if self.use_pyspark and self.spark is not None and not pdf.empty:
            try:
                return self.spark.createDataFrame(pdf)
            except Exception as e:
                logger.warning(f"PySpark dataframe conversion fallback: {e}")
                return pdf
        return pdf

    def stop(self):
        if self.spark:
            self.spark.stop()
