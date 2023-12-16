import os
from data_preprocessing.getData import convert_video_to_frames_and_detect_poses as convert
import tensorflow as tf
import matplotlib.pyplot as plt
from displacement import calculate_displacement
import numpy as np

def getRawDisplacementData(video_path, output_folder='output/'):
    convert(video_path, output_folder)
    # Define the paths
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_folder, video_name, 'csvFile', 'pose_data.csv')
    return calculate_displacement(csv_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the saved model
    loaded_model = tf.keras.models.load_model('pose_detection_model')
    video_path = 'test_data/test_pose.mp4'
    data = getRawDisplacementData(video_path)
    # Predict using the loaded model
    predictions = loaded_model.predict(data)

    # Set a threshold for classification


    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)

    # Check the label
    if np.sum(predicted_labels) > len(predicted_labels) / 2:
        print("The video is classified as a delivery.")
    else:
        print("The video is classified as walking.")

    # Plotting the graph for predictions
    plt.plot(predictions)
    plt.title("Predictions for Video Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Prediction Probability")
    plt.show()

