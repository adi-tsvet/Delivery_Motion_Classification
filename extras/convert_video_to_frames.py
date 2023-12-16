import cv2
import os

def convert_video_to_frames(video_path, output_folder='output/'):
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
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"{frame_count} frames extracted and saved to {output_folder}")

# Example usage
video_path = '../data_preprocessing/sample_input/sampleWalking.mp4'
output_folder = 'output/'
convert_video_to_frames(video_path, output_folder)
