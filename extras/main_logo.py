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
    """
    Detects a logo in a given frame using a sliding window approach.
    Args:
        frame: The frame in which to detect the logo.
        logo_detection_model: The trained model used for logo detection.
        timeout_seconds: Maximum time allowed for the detection process.
    Returns:
        max_prob_frame: The frame with the highest logo probability.
        max_prob: The probability of the logo in the max_prob_frame.
    """
    start_time = time.time()  # Record the start time

    h, w, _ = frame.shape
    stride = 30  # Define the sliding window stride
    window_size = 224  # Define the sliding window size
    stop_threshold = 0.85  # Threshold probability to stop logo detection

    max_prob = 0
    max_prob_frame = None  # Initialize max_prob_frame outside the loop

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            # Check if the timeout has been reached
            if time.time() - start_time > timeout_seconds:
                print("Timeout reached. Stopping logo detection.")
                return max_prob_frame, max_prob, None  # Return the current max_prob_frame

            window = frame[y:y + window_size, x:x + window_size]

            # Preprocess the window for logo detection model
            window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window = cv2.resize(window, (224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(window)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0

            predictions = logo_detection_model.predict(img_array)

            # Iterate over each class label and update if a higher probability is found
            for i, label in enumerate(class_labels):
                if predictions[0, i] > max_prob:  # Check probability for each class
                    max_prob = predictions[0, i]
                    max_prob_frame = frame[y:y + window_size, x:x + window_size]

                    # Check if the probability exceeds the threshold
                    if max_prob > stop_threshold:
                        return max_prob_frame, max_prob, label  # Return the frame, probability, and label

    return max_prob_frame, max_prob, None  # Return the best found frame, probability, and label (if any)

def detect_logo_in_frame(frame, timeout_seconds=20):
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
    stop_threshold = 0.95  # Threshold probability to stop logo detection
    if max_prob > stop_threshold:
        return frame, max_prob, detected_label  # Return the frame, probability, and label

    return frame, max_prob, None  # Return the original frame, probability, and label (if any)


# ... (rest of the code remains unchanged)

# Capture the video
video_path = '../test_data/wallmart_delivery.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Find the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the start frame for the last 10% of the video
start_frame_for_last_segment = int(total_frames * 0.9)

# Generate a random frame number within the last 10% of the video
random_frame_number = random.randint(start_frame_for_last_segment, total_frames - 1)


# # Variables to store frame with the highest logo probability
# max_prob_frame = None
# max_prob = 0.0
# predicted_class = None  # Variable to store the predicted class label

# Set the video frame position to a random frame
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

# Capture the random frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture frame")
    cap.release()
    exit()

# # Focus on the central region of the frame for logo detection
# h, w, _ = frame.shape
# central_region = frame[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]

# Detect logos in the central region
max_prob_frame, max_probs, label = detect_logo_in_frame(frame)

# Draw bounding box around the detected logo area (if found)
if max_prob_frame is not None:
    h, w, _ = max_prob_frame.shape
    cv2.rectangle(max_prob_frame, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 255, 0), 2)
    # Adjust the box size as needed

    # Display the frame with the rectangle and its predicted class
    cv2.imshow(f'Frame with Logo ({label})', max_prob_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No logo found in the randomly selected frame.")

# Release the video capture object
cap.release()