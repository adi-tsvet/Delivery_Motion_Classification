import cv2
from data_preprocessing.pose_constants import BODY_PARTS


def detect_pose(image_path, thr=0.2, width=368, height=368):
    inWidth = width
    inHeight = height

    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    # Read the sample_input image
    input_image = cv2.imread(image_path)

    if input_image is None:
        print("Error: Could not read the sample_input image.")
        return None

    frameWidth = input_image.shape[1]
    frameHeight = input_image.shape[0]

    net.setInput(cv2.dnn.blobFromImage(input_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
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