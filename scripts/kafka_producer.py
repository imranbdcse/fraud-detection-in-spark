"""
Kafka Producer: Credit Card Transaction Simulator
==================================================
Simulates a real-time stream of credit card transactions by generating
random JSON records and publishing them to a Kafka topic.

Usage:
    python scripts/kafka_producer.py [--topic TOPIC] [--broker BROKER]
                                     [--rate RATE] [--fraud-ratio RATIO]
"""

import json
import random
import time
import argparse
import uuid
import logging
from datetime import datetime, timezone

from kafka import KafkaProducer
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

fake = Faker()

# --- Constants ---
MERCHANT_TYPES = [
    "grocery", "online_retail", "restaurant", "electronics",
    "travel", "gas_station", "pharmacy", "clothing", "entertainment",
]

FRAUD_INDICATORS = {
    "high_amount_threshold": 5000.0,
    "unusual_locations": ["Unknown", "Anonymous_IP"],
}


def generate_transaction(fraud_ratio: float = 0.05) -> dict:
    """
    Generate a single simulated credit card transaction.

    Args:
        fraud_ratio: Probability that the generated transaction is fraudulent.

    Returns:
        A dictionary representing a transaction event.
    """
    is_fraud = random.random() < fraud_ratio

    # Fraudulent transactions tend to have higher amounts or unusual patterns
    if is_fraud:
        amount = round(random.uniform(1000.0, 9999.99), 2)
        location = random.choice(FRAUD_INDICATORS["unusual_locations"] + [fake.city()])
        ip_address = fake.ipv4_public()
    else:
        amount = round(random.uniform(1.0, 499.99), 2)
        location = fake.city()
        ip_address = fake.ipv4_private()

    transaction = {
        "TransactionID": str(uuid.uuid4()),
        "UserID": f"user_{random.randint(1000, 9999)}",
        "Timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "TransactionAmount": amount,
        "IPAddress": ip_address,
        "Location": location,
        "MerchantType": random.choice(MERCHANT_TYPES),
        "CardLastFour": str(random.randint(1000, 9999)),
        "TransactionType": random.choice(["purchase", "withdrawal", "refund"]),
        "DeviceID": str(uuid.uuid4()),
        "FraudLabel": int(is_fraud),  # Ground truth (for simulation only)
    }
    return transaction


def create_producer(broker: str) -> KafkaProducer:
    """
    Create and return a KafkaProducer connected to the specified broker.

    Args:
        broker: Kafka broker address (e.g., 'localhost:9092').

    Returns:
        A configured KafkaProducer instance.
    """
    producer = KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )
    logger.info("Connected to Kafka broker at %s", broker)
    return producer


def run_producer(
    topic: str,
    broker: str,
    rate: float,
    fraud_ratio: float,
    max_messages: int = 0,
) -> None:
    """
    Continuously generate and send transaction events to Kafka.

    Args:
        topic:        Kafka topic name to produce messages to.
        broker:       Kafka broker address.
        rate:         Messages per second to produce.
        fraud_ratio:  Fraction of messages that simulate fraud.
        max_messages: Stop after this many messages (0 = run indefinitely).
    """
    producer = create_producer(broker)
    interval = 1.0 / rate if rate > 0 else 1.0
    count = 0

    logger.info(
        "Starting producer → topic='%s', rate=%.1f msg/s, fraud_ratio=%.2f",
        topic, rate, fraud_ratio,
    )

    try:
        while True:
            transaction = generate_transaction(fraud_ratio)
            producer.send(topic, value=transaction)
            count += 1

            if transaction["FraudLabel"] == 1:
                logger.info("[FRAUD]   Sent transaction %s", transaction["TransactionID"])
            else:
                logger.debug("[NORMAL]  Sent transaction %s", transaction["TransactionID"])

            if max_messages > 0 and count >= max_messages:
                logger.info("Reached max_messages=%d. Stopping.", max_messages)
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        logger.info("Producer closed. Total messages sent: %d", count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate credit card transaction streams to Kafka."
    )
    parser.add_argument(
        "--topic", default="transactions", help="Kafka topic name (default: transactions)"
    )
    parser.add_argument(
        "--broker", default="localhost:9092", help="Kafka broker address (default: localhost:9092)"
    )
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Messages per second (default: 1.0)"
    )
    parser.add_argument(
        "--fraud-ratio", type=float, default=0.05,
        help="Fraction of transactions that are fraudulent (default: 0.05)"
    )
    parser.add_argument(
        "--max-messages", type=int, default=0,
        help="Stop after this many messages; 0 = run indefinitely (default: 0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_producer(
        topic=args.topic,
        broker=args.broker,
        rate=args.rate,
        fraud_ratio=args.fraud_ratio,
        max_messages=args.max_messages,
    )
