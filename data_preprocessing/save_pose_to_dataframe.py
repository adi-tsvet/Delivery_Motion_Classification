import pandas as pd
import os
from data_preprocessing.detect_pose import detect_pose
from data_preprocessing.pose_constants import BODY_COLUMNS, BODY_PARTS
import numpy as np

def save_dataframe(image_path, output_folder, thr=0.2, width=368, height=368):

    pose_points = detect_pose(image_path, thr, width, height)

    if pose_points is not None:
        # Convert pose points to DataFrame
        data = []

        for point in pose_points:
            if point:
                x, y = point
                confidence = 1.0  # You can modify this based on your use case
                data.extend([x, y])
            else:
                data.extend([0, 0])

        data_array = np.array(data).reshape(1, -1)
        # Save the numpy array
        npy_folder = os.path.join(output_folder, 'npyFile')
        os.makedirs(npy_folder, exist_ok=True)

        output_npy_path = os.path.join(npy_folder, 'pose_data.npy')

        np.save(output_npy_path, data_array)
        df = pd.DataFrame(data_array, columns=BODY_COLUMNS)

        csv_folder = os.path.join(output_folder, 'csvFile')
        os.makedirs(csv_folder, exist_ok=True)

        # Define the output_csv_path before using it
        output_csv_path = os.path.join(csv_folder, 'pose_data.csv')

        # Save DataFrame to a CSV file in append mode with header
        df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)