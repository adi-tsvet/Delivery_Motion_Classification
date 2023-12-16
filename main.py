import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import random
import time
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image
import os
from data_preprocessing.getData import convert_video_to_frames_and_detect_poses as convert
import tensorflow as tf
import matplotlib.pyplot as plt
from displacement import calculate_displacement
import numpy as np

# Load models
delivery_classification_model = load_model('pose_detection_model')
logo_detection_model = load_model('logo_detection/logo_detection_model')

class_labels = ['Amazon', 'Fedex', 'Usps', 'Walmart']

def load_and_preprocess_image(image_path):
    # Load the image using PIL
    img = Image.open(image_path)

    # Convert to RGB (in case it's a grayscale image)
    img = img.convert('RGB')

    # Resize the image to match the input size expected by the model
    input_size = (224, 224)  # Adjust this size based on your model's input requirements
    img = img.resize(input_size)

    # Convert the image to an array and preprocess for the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Normalize pixel values to be in the range [0, 1]
    img_array = img_array / 255.0

    return img_array
def detect_logo_in_frame(frame, timeout_seconds=20):
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
def getRawDisplacementData(video_path, output_folder='output/'):
    convert(video_path, output_folder)
    # Define the paths
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_folder, video_name, 'csvFile', 'pose_data.csv')
    return calculate_displacement(csv_path),output_folder

def extract_frames(output_folder):
    frame_paths = []
    for subdir, dirs, files in os.walk(output_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                frame_paths.append(os.path.join(subdir, file))

    selected_frames = frame_paths[::50]  # Select every 50th frame
    return [cv2.imread(frame) for frame in selected_frames]

def classify_video_and_detect_logo(video_path):
    data,output_folder = getRawDisplacementData(video_path)
    # Predict using the loaded model
    predictions = delivery_classification_model.predict(data)

    # Set a threshold for classification

    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    # Plotting the graph for predictions
    plt.plot(predictions)
    plt.title("Predictions for Video Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Prediction Probability")
    plt.show()
    # Check the label
    is_delivery = np.sum(predicted_labels) > len(predicted_labels) / 2


    if is_delivery:
        print("The video is classified as a delivery.")
        # Extract frames from the video or use saved frames
        frames = extract_frames(output_folder)

        for frame in frames:
            # Detect logo in the frame
            logo_frame, logo_prob, logo_label = detect_logo_in_frame(frame)

            if logo_frame is not None and logo_prob >= threshold:
                # Found a logo, you can break or process further
                print(f"Logo detected: {logo_label} with probability {logo_prob}")
                # Display or save the frame with detected logo
                cv2.imshow("Detected Logo", logo_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
    else:
        print("The video is classified as walking.")

# Run the combined process on a video
video_path = 'test_data/test_pose.mp4'
classify_video_and_detect_logo(video_path)
