import cv2
import os
import pandas as pd

from detect_pose_in_frame import detect_pose


def detect_poses_and_save(video_path, output_folder='output/', thr=0.2, width=368, height=368):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the frames per second (fps) and frame dimensions
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize an empty DataFrame to store pose data
    columns = ["Frame", "BodyPart", "X", "Y", "Confidence"]
    pose_df = pd.DataFrame(columns=columns)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        # Detect pose in the frame and save pose data to DataFrame
        pose_points = detect_pose(frame_path, thr, width, height)
        if pose_points:
            frame_data = []
            for body_part, point in zip(BODY_PARTS.keys(), pose_points):
                if point:
                    x, y = point
                    confidence = 1.0  # You can modify this based on your use case
                    frame_data.append([frame_count, body_part, x, y, confidence])

            frame_df = pd.DataFrame(frame_data, columns=columns)
            pose_df = pd.concat([pose_df, frame_df], ignore_index=True)

        frame_count += 1

    # Release the video capture object
    cap.release()

    # Save pose data to a CSV file
    output_csv_path = os.path.join(output_folder, 'pose_data.csv')
    pose_df.to_csv(output_csv_path, index=False)
    print(f"{frame_count} frames extracted and pose data saved to: {output_csv_path}")

# Call the function with the video file path
video_path = '../data_preprocessing/sample_input/sampleWalking.mp4'
detect_poses_and_save(video_path)
