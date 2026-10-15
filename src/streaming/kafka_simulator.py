"""
Kafka Streaming Ingestion & Simulation Module
Implements Section 3.4 of Susi & Shanthi (2023): "Static Tweets Simulation"

Provides:
1. Native Kafka Producer/Consumer via kafka-python-ng for production environments.
2. High-performance SimulatedKafkaBroker for standalone/offline execution without external cluster dependencies.
"""

import json
import time
import queue
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulatedKafkaBroker:
    """
    In-memory Kafka broker simulator supporting topics, partitions,
    asynchronous producer publishing, and window-based consumer polling.
    """
    def __init__(self):
        self.topics: Dict[str, queue.Queue] = {}
        self.message_logs: Dict[str, List[Dict[str, Any]]] = {}

    def create_topic(self, topic_name: str):
        if topic_name not in self.topics:
            self.topics[topic_name] = queue.Queue()
            self.message_logs[topic_name] = []
            logger.info(f"Topic '{topic_name}' created in Kafka broker.")

    def produce(self, topic_name: str, message: Dict[str, Any]):
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        payload = {
            'offset': len(self.message_logs[topic_name]),
            'timestamp': datetime.now().isoformat(),
            'value': message
        }
        self.topics[topic_name].put(payload)
        self.message_logs[topic_name].append(payload)

    def consume_batch(self, topic_name: str, batch_size: int = 50, timeout: float = 0.5) -> List[Dict[str, Any]]:
        if topic_name not in self.topics:
            return []
        
        q = self.topics[topic_name]
        batch = []
        start_time = time.time()

        while len(batch) < batch_size and (time.time() - start_time) < timeout:
            try:
                msg = q.get(block=True, timeout=0.05)
                batch.append(msg['value'])
                q.task_done()
            except queue.Empty:
                break

        return batch

    def get_total_messages(self, topic_name: str) -> int:
        return len(self.message_logs.get(topic_name, []))


class TweetStreamPublisher:
    """
    Kafka Producer for replaying static historical tweets as a continuous real-time data stream.
    """
    def __init__(self, broker: Optional[SimulatedKafkaBroker] = None, kafka_bootstrap_servers: Optional[str] = None):
        self.kafka_servers = kafka_bootstrap_servers
        self.native_producer = None
        self.sim_broker = broker or SimulatedKafkaBroker()

        if self.kafka_servers:
            try:
                from kafka import KafkaProducer
                self.native_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info(f"Connected to real Kafka cluster at {self.kafka_servers}")
            except Exception as e:
                logger.warning(f"Could not connect to Kafka at {self.kafka_servers}: {e}. Falling back to Simulated Broker.")
                self.native_producer = None

    def publish_tweet(self, topic: str, tweet_record: Dict[str, Any]):
        if self.native_producer:
            self.native_producer.send(topic, tweet_record)
        else:
            self.sim_broker.produce(topic, tweet_record)

    def publish_stream(self, topic: str, tweets: List[Dict[str, Any]], delay_sec: float = 0.0):
        for tweet in tweets:
            self.publish_tweet(topic, tweet)
            if delay_sec > 0:
                time.sleep(delay_sec)


class TweetStreamConsumer:
    """
    Kafka Consumer to ingest streaming tweets in window chunks.
    """
    def __init__(self, topic: str, broker: SimulatedKafkaBroker, kafka_bootstrap_servers: Optional[str] = None):
        self.topic = topic
        self.sim_broker = broker
        self.kafka_servers = kafka_bootstrap_servers
        self.native_consumer = None

        if self.kafka_servers:
            try:
                from kafka import KafkaConsumer
                self.native_consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.kafka_servers,
                    auto_offset_reset='earliest',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )
            except Exception as e:
                logger.warning(f"Native KafkaConsumer init failed: {e}. Using simulated broker.")
                self.native_consumer = None

    def poll_window(self, window_size: int = 100, timeout: float = 1.0) -> List[Dict[str, Any]]:
        if self.native_consumer:
            # Poll from native kafka
            records = []
            msg_dict = self.native_consumer.poll(timeout_ms=int(timeout * 1000), max_records=window_size)
            for tp, messages in msg_dict.items():
                for m in messages:
                    records.append(m.value)
            return records
        else:
            return self.sim_broker.consume_batch(self.topic, batch_size=window_size, timeout=timeout)
