import cv2 as cv
import numpy as np
import argparse
import os
import pandas as pd

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
def detect_pose(image_path, thr=0.2, width=368, height=368):
    inWidth = width
    inHeight = height

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    # Read the sample_input image
    input_image = cv.imread(image_path)

    if input_image is None:
        print("Error: Could not read the sample_input image.")
        return None

    frameWidth = input_image.shape[1]
    frameHeight = input_image.shape[0]

    net.setInput(cv.dnn.blobFromImage(input_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    return points

def save_pose_to_dataframe(image_path, output_folder, thr=0.2, width=368, height=368):
    # Detect pose points
    pose_points = detect_pose(image_path, thr, width, height)

    if pose_points is not None:
        # Convert pose points to DataFrame
        columns = ["BodyPart", "X", "Y", "Confidence"]
        data = []

        for body_part, point in zip(BODY_PARTS.keys(), pose_points):
            if point:
                x, y = point
                confidence = 1.0  # You can modify this based on your use case
                data.append([body_part, x, y, confidence])

        df = pd.DataFrame(data, columns=columns)

        # Save DataFrame to a CSV file
        output_csv_path = os.path.join(output_folder, 'pose_data.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"Pose data saved to: {output_csv_path}")

# Example usage
image_path = '../data_preprocessing/sample_input/image.jpg'
output_folder = 'output/'
save_pose_to_dataframe(image_path, output_folder)
