import numpy as np
import json
from confluent_kafka import Consumer, KafkaException, KafkaError

# --- Configuration ---
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'image-processor-group',
    'auto.offset.reset': 'earliest',
    # Increase the session timeout. If reassembling takes a long time,
    # the consumer might be considered dead by the group coordinator.
    'session.timeout.ms': 30000, 
    #'fetch.message.max.bytes': 10485760
}

consumer = Consumer(conf)
topic = 'image-topic'
consumer.subscribe([topic])

# In-memory buffer to reassemble chunks
incomplete_chunks = {}

def process_fully_reassembled_image(image_id, image_bytes):
    """
    This function is called when an image is fully reassembled.
    """
    print(f"\nSUCCESS: Image {image_id} fully reassembled.")
    print(f"Total size: {len(image_bytes) / (1024*1024):.2f} MB")
    
    # For demonstration, convert bytes back to a numpy array and check its shape
    try:
        reconstructed_array = np.frombuffer(image_bytes, dtype=np.int32).reshape((8120, 8120))
        print(f"Image reconstructed into numpy array with shape: {reconstructed_array.shape}")
        # Here you would do your actual image processing (e.g., save to disk, analyze, etc.)
        # For example:
        # from PIL import Image
        # img = Image.fromarray((reconstructed_array * 255).astype(np.uint8))
        # img.save(f"{image_id}.png")

    except Exception as e:
        print(f"Error processing reassembled image {image_id}: {e}")

# --- Main execution ---
try:
    print(f"Subscribed to topic '{topic}'. Waiting for messages...")
    while True:
        msg = consumer.poll(timeout=1.0)
        
        if msg is None:
            continue
            
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event, not an error
                continue
            else:
                raise KafkaException(msg.error())

        # 1. Decode the key to get chunk metadata
        key_data = json.loads(msg.key().decode('utf-8'))
        image_id = key_data['image_id']
        index = key_data['index']
        total_chunks = key_data['total_chunks']

        #print(f"Received chunk {index + 1}/{total_chunks} for image {image_id}")
        
        # 2. Store the chunk data
        if image_id not in incomplete_chunks:
            # Initialize storage for this new image
            incomplete_chunks[image_id] = {
                'total_chunks': total_chunks,
                'chunks': [None] * total_chunks # Use a list for ordered insertion
            }
        
        # Store the chunk in the correct position
        incomplete_chunks[image_id]['chunks'][index] = msg.value()
        
        # 3. Check if all chunks for this image have arrived
        # `all()` is a fast way to check if no `None` values are left
        if all(chunk is not None for chunk in incomplete_chunks[image_id]['chunks']):
            
            # 4. Reassemble the image
            # Join all the byte chunks together
            full_image_bytes = b''.join(incomplete_chunks[image_id]['chunks'])
            
            # Process the complete image data
            process_fully_reassembled_image(image_id, full_image_bytes)
            
            # 5. Clean up the buffer to free memory
            del incomplete_chunks[image_id]
            print(f"Cleaned up buffer for image {image_id}.")

except KeyboardInterrupt:
    print("Aborted by user.")
finally:
    # Close down consumer to commit final offsets.
    consumer.close()
    print("Consumer closed.")