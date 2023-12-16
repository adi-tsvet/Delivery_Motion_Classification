import cv2
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from data_preprocessing.pose_constants import BODY_PARTS

def detect_pose(frame, thr=0.2, width=368, height=368):
    inWidth = width
    inHeight = height

    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    if frame is None:
        print("Error: Received an empty frame.")
        return None

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    return points

# Load the trained model
model = load_model('motion_detection_model')  # Load the previously saved model

# Open the webcam
cap = cv2.VideoCapture(0)  # Change to the appropriate index if using multiple cameras

# Set the frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Motion label mapping
label_mapping = {0: 'standing', 1: 'walking'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect pose from the frame
    pose_points = detect_pose(frame)  # Assuming detect_pose directly operates on the frame

    # Process pose points to match the model's sample_input shape
    processed_data = []

    # Convert pose points to match the expected sample_input shape of the model
    for point in pose_points:
        if point:
            x, y = point
            processed_data.extend([x, y])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Change color or size as needed
        else:
            processed_data.extend([0, 0])

    # Convert to numpy array and reshape to match model's sample_input shape
    processed_data_array = np.array(processed_data).reshape(1, -1)

    # Make predictions using the processed data
    predictions = model.predict(processed_data_array)

    # Apply threshold to get binary labels
    threshold = 0.5  # You can use the threshold determined during training or adjust as needed
    binary_labels = (predictions > threshold).astype(int)

    # Map binary labels to motion labels
    labels_mapped = np.vectorize(label_mapping.get)(binary_labels)

    # Add motion label text to the frame
    text = f"Motion: {labels_mapped[0]}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the frame with motion label
    cv2.imshow('Motion Detection', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
