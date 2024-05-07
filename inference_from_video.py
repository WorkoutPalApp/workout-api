import joblib

from inference import preprocess_frame
import pandas as pd
import cv2


# Script for displaying predictions live on the screen (used for demo)

# Deadlift/Users/andycraig/PycharmProjects/pythonProject/bilbo_swaggins181/2024-01-03_00-32-13_UTC.mp4
# SQ /Users/andycraig/PycharmProjects/pythonProject/bilbo_swaggins181/2024-01-06_22-04-20_UTC_1.mp4
def draw_keypoints(image, prediction, key_points, bounding_box):
    # Step 1: Calculate the width and height of the bounding box
    cv2.putText(image, f"Prediction: {prediction}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8)

    x1, y1, x2, y2 = bounding_box
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    box_width = x2 - x1
    box_height = y2 - y1
    image_width, image_height = 480, 480
    cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
    relative_positions = []

    # Draw key points within the bounding box
    for kp in key_points:
        kp_x, kp_y = kp
        kp_x = int(kp_x)
        kp_y = int(kp_y)
        cv2.circle(image, (kp_x, kp_y), 5, (0, 255, 0), -1)

    # Display image
    cv2.imshow("Bounding Box with Key Points", image)
    cv2.waitKey(1)  # Adjust delay time as needed
    return relative_positions


video_path = '/Users/andycraig/PycharmProjects/pythonProject/bilbo_swaggins181/2024-01-03_00-32-13_UTC.mp4'

# Open the video file
cap = cv2.VideoCapture(0)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file")
    exit()
columns = ['box_width', 'box_height']
for i in range(17):
    columns.append(f'kp_{i + 1}_x')
    columns.append(f'kp_{i + 1}_y')
# Loop through each frame
model_path = 'random_forest_model_v2.pkl'
rf_model = joblib.load(model_path)

# main loop
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("hi")
        break

    try:
        # Preprocess the frame for prediction (yolo model)
        results, data = preprocess_frame(frame)
        # Make model input df
        results_df = pd.DataFrame([data], columns=columns)
        # Make exercise prediction
        prediction = rf_model.predict(results_df)
        # Extract Key points and bounding box from YOLOv8 output
        kps = results[0].keypoints.xy.numpy()[0]
        bb = results[0].boxes.xyxy.numpy()[0]
        draw_keypoints(frame, prediction, kps, bb)
    except IndexError as e:
        results = None

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
