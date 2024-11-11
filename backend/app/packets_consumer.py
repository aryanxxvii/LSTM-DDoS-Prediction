import torch
import torch.nn as nn
import json
import logging
import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_TOPIC = 'packet-topic'
KAFKA_BROKER = 'kafka:9092'
GROUP_ID = 'packet-consumer-group'
MAX_RETRIES = 10
RETRY_DELAY = 5
MODEL_PATH = "/app/model/model_4lstm.pth"
PREDICTION_TOPIC = "prediction-topic"

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(34, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.lstm4 = nn.LSTM(64, 32, batch_first=True)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

def predict(model, data, device):
    try:
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = model(data_tensor)
        return prediction.item()
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None

def connect_kafka_consumer(retries=MAX_RETRIES):
    for i in range(retries):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                group_id=GROUP_ID,
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info("Successfully connected to Kafka")
            return consumer
        except Exception as e:
            if i == retries - 1:
                logger.error(f"Failed to connect to Kafka after {retries} attempts")
                raise
            logger.warning(f"Failed to connect to Kafka, attempt {i + 1}/{retries}")
            time.sleep(RETRY_DELAY)

def publish_predictions(prediction):
    try:
        prediction_data = {
            'timestamp': time.time(),
            'prediction': prediction,
            'is_ddos': bool(prediction > 0.5)
        }
        logger.info(f"Prediction sent: {prediction_data}")
    except Exception as e:
        logger.error(f"Error publishing prediction: {str(e)}")

def main():
    try:
        model, device = load_model()
        logger.info(f"Model loaded successfully. Using device: {device}")

        consumer = connect_kafka_consumer()
        

        logger.info("Starting to process messages...")
        
        for message in consumer:
            try:
                data = message.value
                sequence = data.get('sequence')
                
                if sequence:
                    sequence = sequence.unsqueeze(0)
                    prediction = predict(model, sequence, device)
                    if prediction is not None:
                        publish_predictions(prediction)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                continue

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if 'consumer' in locals():
            consumer.close()

if __name__ == "__main__":
    main()