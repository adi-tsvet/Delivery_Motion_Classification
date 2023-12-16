import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import random
import time


# Load the logo detection model
logo_detection_model = load_model('logo_detection/logo_detection_model')

class_labels = ['Amazon', 'Fedex', 'Usps', 'Walmart']

def detect_logo_in_frame_with_stride(frame, timeout_seconds=20):
    start_time = time.time()
    h, w, _ = frame.shape
    stride = 100  # Smaller stride for finer scanning
    window_sizes = [200, 300, 400]  # Different window sizes for multi-scale detection

    max_prob = 0
    max_prob_frame = None

    for window_size in window_sizes:
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                if time.time() - start_time > timeout_seconds:
                    print("Timeout reached. Stopping logo detection.")
                    return max_prob_frame, max_prob, None

                window = frame[y:y + window_size, x:x + window_size]
                window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                window = cv2.resize(window, (224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(window)
                img_array = tf.expand_dims(img_array, 0)
                img_array = img_array / 255.0

                predictions = logo_detection_model.predict(img_array)

                for i, label in enumerate(class_labels):
                    if predictions[0, i] > max_prob:
                        max_prob = predictions[0, i]
                        max_prob_frame = frame[y:y + window_size, x:x + window_size]

                        if max_prob > 0.8:
                            return max_prob_frame, max_prob, label

    return max_prob_frame, max_prob, None


def detect_logo_in_frame(frame, timeout_seconds=40):
    """
    Detects a logo in the given frame.
    Args:
        frame: The frame in which to detect the logo.
        logo_detection_model: The trained model used for logo detection.
        timeout_seconds: Maximum time allowed for the detection process.
    Returns:
        frame: The original frame (not cropped or resized).
        max_prob: The probability of the logo in the frame.
        detected_label: The label of the detected logo (if any).
    """
    start_time = time.time()  # Record the start time

    # Resize the entire frame to the expected input size of the model
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_frame)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    # Check if the timeout has been reached
    if time.time() - start_time > timeout_seconds:
        print("Timeout reached. Stopping logo detection.")
        return frame, 0, None  # Return the original frame with no detection

    predictions = logo_detection_model.predict(img_array)

    # Find the maximum probability and its corresponding label
    max_prob = max(predictions[0])
    max_prob_index = tf.argmax(predictions[0]).numpy()
    detected_label = class_labels[max_prob_index]

    # Check if the probability exceeds the threshold
    stop_threshold = 0.5  # Threshold probability to stop logo detection
    if max_prob > stop_threshold:
        return frame, max_prob, detected_label  # Return the frame, probability, and label

    return frame, max_prob, None  # Return the original frame, probability, and label (if any)


# Capture the video
video_path = 'test_data/wallmart_delivery.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Find the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Iterate through frames at intervals of 40
for frame_idx in range(0, total_frames, 40):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Capture the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to capture frame at index {frame_idx}")
        continue

    # Detect logos in the frame
    max_prob_frame, max_prob, label = detect_logo_in_frame_with_stride(frame)

    # Check if a logo is detected with high enough probability
    if max_prob_frame is not None and max_prob>=.8:
        h, w, _ = max_prob_frame.shape
        cv2.rectangle(max_prob_frame, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 255, 0), 2)
        # Adjust the box size as needed
        print(f'Frame with Logo ({label})')
        # Display the frame with the rectangle and its predicted class
        cv2.imshow(f'Frame with Logo ({label})', max_prob_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break  # Break the loop if a logo is found
    else:
        print(f"No logo found in frame at index {frame_idx}")

# Release the video capture object
cap.release()