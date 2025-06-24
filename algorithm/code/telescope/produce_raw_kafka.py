# # kafka_publisher.py
# from confluent_kafka import Producer

# conf = {
#      'bootstrap.servers': '192.168.1.101:9092,192.168.1.105:9092192.168.1.106:9092,192.168.1.107:9092'
# }
# producer = Producer(conf)

# topic = 'test-topic'

# def delivery_report(err, msg):
#     if err is not None:
#         print(f"Delivery failed: {err}")
#     else:
#         print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# for i in range(10):
#     message = f"Hello Kafka {i}"
#     producer.produce(topic, message.encode('utf-8'), callback=delivery_report)
#     producer.poll(0)

# producer.flush()

import numpy as np
import uuid
import json
import base64
import math
from confluent_kafka import Producer

# --- Configuration ---
conf = {
    # NOTE: There is a typo in your original bootstrap.servers list
    'bootstrap.servers': 'localhost:9092',
    # Optional: Increase linger.ms and batch.size for better throughput with many small messages
    'linger.ms': 10000,
    'batch.size': 131072, # 128KB
    'fetch.message.max.bytes': 10485760 # 10MB, this is the maximum size of a message that can be fetched
}

producer = Producer(conf)
topic = 'image-topic' # Use a new topic for clarity

# --- Chunking Configuration ---
# Kafka's default is 1MB. We'll use 512KB to be safe.
CHUNK_SIZE_KB = int(512*1.9)
CHUNK_SIZE_BYTES = CHUNK_SIZE_KB * 1024

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        print(f"Delivery failed for message {msg.key()}: {err}")
    else:
        # Getting the key from the message metadata
        key = json.loads(msg.key().decode('utf-8'))
        #print(f"Chunk {key['index'] + 1}/{key['total_chunks']} for image {key['image_id']} delivered to {msg.topic()} [{msg.partition()}]")

def send_image_as_chunks(image_data, topic):
    """
    Splits the image data into chunks and sends them to Kafka.
    """
    image_id = str(uuid.uuid4())
    total_bytes = len(image_data)
    total_chunks = math.ceil(total_bytes / CHUNK_SIZE_BYTES)
    
    print(f"\nSending image {image_id}...")
    print(f"Total size: {total_bytes / (1024*1024):.2f} MB, Total chunks: {total_chunks}")

    for i in range(total_chunks):
        start_byte = i * CHUNK_SIZE_BYTES
        end_byte = min((i + 1) * CHUNK_SIZE_BYTES, total_bytes)
        chunk_data = image_data[start_byte:end_byte]
        
        # We use a key to ensure all chunks of the same image go to the same partition,
        # which helps with in-order processing.
        # The message value contains the actual chunk data.
        message_key = {
            'image_id': image_id,
            'index': i,
            'total_chunks': total_chunks
        }
        
        # Serialize key to JSON string
        key_str = json.dumps(message_key).encode('utf-8')
        
        producer.produce(
            topic, 
            value=chunk_data, 
            key=key_str, 
            callback=delivery_report
        )
        # Poll to allow the producer to send buffered messages
        producer.poll(0)

    print(f"All {total_chunks} chunks for image {image_id} have been produced.")

# --- Main execution ---
if __name__ == "__main__":
    # 1. Create a dummy 8k * 8k 32-bit (float32) image
    print("Creating a dummy 8k*8k 32-bit image (256MB)...")
    # A float32 is 4 bytes, which is 32 bits
    dummy_image_array = np.random.rand(8120, 8120).astype(np.int32)
    
    # 2. Convert numpy array to raw bytes for transport
    image_bytes = dummy_image_array.tobytes()

    # 3. Send the image
    send_image_as_chunks(image_bytes, topic)
    
    # 4. Wait for all messages in the producer queue to be delivered.
    print("\nFlushing producer...")
    producer.flush()
    print("Producer flushed. All messages sent.")