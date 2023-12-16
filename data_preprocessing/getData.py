import os
import cv2
from data_preprocessing.save_pose_to_dataframe import save_dataframe



def convert_video_to_frames_and_detect_poses(video_path, output_folder='output/'):
    # Extract video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the output folder with the video file name if it doesn't exist
    output_folder = os.path.join(output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

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

    frame_count = 0
    print("Processsing ...")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)


        # Detect pose in the frame and save pose data to DataFrame
        save_dataframe(frame_path, output_folder)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"{frame_count} frames extracted and pose data saved to {output_folder}")
    print(f"Pose data appended to: {output_folder}")

# video_path = 'sample_input/sampleStanding.mp4'
#
# folder_path = 'prepared_input/'  # Replace this with your folder's path
# mp4_files = [file for file in os.listdir(folder_path) if file.endswith('.mp4')]
# output_folder = 'prepared_output/'
#
# for i in mp4_files:
#     convert_video_to_frames_and_detect_poses(folder_path + i, output_folder)