import time
import pickle
import json
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_TOPIC = 'packet-topic'
KAFKA_BROKER = 'kafka:9092'
DATA_PATH = '/app/data/test/testX.pkl'

def load_test_data():
    """Load and prepare test data."""
    logger.info(f"Loading test data from {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        testX = pickle.load(f)
    testX_np = testX.reshape(-1, 10, 34)
    logger.info(f"Loaded {testX_np.shape[0]} sequences of length {testX_np.shape[1]} with {testX_np.shape[2]} features")
    return testX_np

def create_producer():
    """Create Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        logger.info("Kafka Producer created successfully.")
        return producer
    except Exception as e:
        logger.error(f"Error creating Kafka producer: {str(e)}")
        raise

def main():
    """Main function to simulate packet flow and send data sequence by sequence."""
    logger.info("Waiting for Kafka to be ready...")
    time.sleep(10)
    
    testX_np = load_test_data()

    producer = create_producer()

    logger.info("Starting packet simulation...")

    try:
        for i, sequence in enumerate(testX_np, 1):
            record = {"sequence": sequence.tolist()}

            producer.send(KAFKA_TOPIC, value=record)
            logger.info(f"Sent sequence {record}")

            time.sleep(1)

        producer.flush()
        logger.info(f"Completed sending {i} sequences.")
    
    except KeyboardInterrupt:
        logger.info("Packet simulation stopped.")
    except Exception as e:
        logger.error(f"Error during packet production: {str(e)}")
        raise
    finally:
        producer.close()

if __name__ == "__main__":
    main()
