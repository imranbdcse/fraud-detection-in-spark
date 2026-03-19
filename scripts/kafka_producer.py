"""
Kafka producer that simulates real-time credit card transactions.

Sends JSON-encoded transaction messages to the ``transactions`` Kafka topic
at a configurable rate.

Usage
-----
    python scripts/kafka_producer.py
    python scripts/kafka_producer.py --rate 5 --topic my-transactions
"""

import argparse
import json
import logging
import random
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
MERCHANT_TYPES = ["online", "retail", "restaurant", "grocery", "travel", "entertainment"]


def generate_transaction(rng: random.Random) -> dict:
    """Generate a single synthetic transaction record."""
    features = {f"V{i}": round(rng.gauss(0, 1), 6) for i in range(1, 29)}
    return {
        "TransactionID": f"TXN{rng.randint(100000, 999999)}",
        "UserID": f"USER{rng.randint(1, 1000):04d}",
        "Timestamp": time.time(),
        "Amount": round(rng.lognormvariate(3.5, 1.5), 2),
        "Location": rng.choice(LOCATIONS),
        "MerchantType": rng.choice(MERCHANT_TYPES),
        **features,
        "Time": time.time() % 172800,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Send synthetic transactions to Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                        help="Kafka bootstrap servers.")
    parser.add_argument("--topic", default="transactions",
                        help="Kafka topic to produce to.")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Transactions per second (default: 1).")
    parser.add_argument("--count", type=int, default=0,
                        help="Stop after this many messages (0 = run forever).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.error("kafka-python is not installed. Run: pip install kafka-python")
        sys.exit(1)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    rng = random.Random(args.seed)
    delay = 1.0 / max(args.rate, 0.001)
    sent = 0

    logger.info("Producing to topic '%s' on %s at %.1f msg/s …",
                args.topic, args.bootstrap_servers, args.rate)

    try:
        while args.count == 0 or sent < args.count:
            tx = generate_transaction(rng)
            producer.send(args.topic, tx)
            sent += 1
            if sent % 100 == 0:
                logger.info("Sent %d transactions", sent)
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Producer stopped by user after %d messages.", sent)
    finally:
        producer.flush()
        producer.close()
        logger.info("Total messages sent: %d", sent)


if __name__ == "__main__":
    main()
