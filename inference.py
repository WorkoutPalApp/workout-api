import joblib
from ultralytics import YOLO
import cv2
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import mode


# Function for preprocessing a frame
def preprocess_frame(frame):
    # Example preprocessing steps (you may need to adjust based on your model requirements)
    body_model = YOLO("yolov8n-pose.pt")
    results = body_model.predict(frame, verbose=False)
    data = calculate_rebased_keypoints(key_points=results[0].keypoints.xyn.numpy()[0],
                                       bounding_box=results[0].boxes.xyxyn.numpy()[0])
    return results, data


def calculate_rebased_keypoints(key_points, bounding_box):
    x1, y1, x2, y2 = bounding_box
    box_width = x2 - x1
    box_height = y2 - y1
    data = [box_width, box_height]

    for kp in key_points:
        x_norm, y_norm = kp
        x_rebased_n = ((x_norm - x1) / box_width)
        y_rebased_n = ((y_norm - y1) / box_height)
        data.extend([x_rebased_n, y_rebased_n])
    return data


def generate_prediction_metadata_from_video(video_path, skip_frames, columns):
    # Check if the provided path is a directory
    # Load the trained Random Forest model
    model_path = 'random_forest_model_v2.pkl'  # Update with the path to your trained model
    rf_model = joblib.load(model_path)

    body_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file")
        exit()
    metadata = []
    predictions = []
    count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break
        count += 1
        if count % skip_frames == 0:
            try:
                kp_results = body_model.predict(frame, verbose=False)
                row = calculate_rebased_keypoints(key_points=kp_results[0].keypoints.xyn.numpy()[0],
                                                  bounding_box=kp_results[0].boxes.xyxyn.numpy()[0])

                results_df = pd.DataFrame([row], columns=columns)
                metadata.append(row)
                prediction = rf_model.predict(results_df)
                predictions.append(prediction)
            except Exception as e:
                # handle other exceptions
                print(f'An error occurred when processing frame #{count}')
    metadata_df = pd.DataFrame(metadata, columns=columns)
    predictions_df = pd.DataFrame(predictions, columns=['exercise'])
    metadata_df = metadata_df.join(predictions_df)
    # Write DataFrame to CSV file
    return metadata_df


def get_repetitions_from_exercise(metadata, frame_rate, kernel_size, threshold_multiplier, threshold_peak):
    # Smooth Data
    metadata = medfilt(metadata['box_height'], kernel_size=kernel_size)
    # Find the most common value
    most_common_value = (mode(metadata)[0]) - threshold_peak
    # Define threshold as 70% of the most common value
    threshold = threshold_multiplier * most_common_value
    # Initialize variables
    num_squats = 0
    below_threshold = False
    reps = []
    active_duration = 0
    # Iterate through the squat height data
    for i, height in enumerate(metadata):
        if height < threshold and not below_threshold:
            below_threshold = True
        elif height >= threshold and below_threshold:
            num_squats += 1
            b = i
            f = i
            prev_height = metadata[b]
            next_height = metadata[f]
            while prev_height <= most_common_value:
                b = b - 1
                prev_height = metadata[b]
            while next_height <= most_common_value:
                f = f + 1
                next_height = metadata[f]

            below_threshold = False
            rep = {"start_time": b / frame_rate,
                   "end_time": f / frame_rate,
                   "elapsed_time": ((f - b) / frame_rate)}
            reps.append(rep)
            active_duration += rep["elapsed_time"]
    avg_rep_time = active_duration / num_squats
    return metadata, active_duration, avg_rep_time, reps


def get_repetitions(exercise, metadata, frame_rate):
    if exercise == "Squat":
        kernel_size = 9
        threshold_multiplier = 0.7
        threshold_peak = 0.01
        return get_repetitions_from_exercise(metadata=metadata, frame_rate=frame_rate, kernel_size=kernel_size, threshold_multiplier=threshold_multiplier, threshold_peak=threshold_peak)
    if exercise == "Deadlift":
        kernel_size = 9
        threshold_multiplier = 0.7
        threshold_peak = 0.01
        return get_repetitions_from_exercise(metadata=metadata, frame_rate=frame_rate, kernel_size=kernel_size,
                                             threshold_multiplier=threshold_multiplier, threshold_peak=threshold_peak)
    if exercise == "Pushup":
        kernel_size = 9
        threshold_multiplier = 0.7
        threshold_peak = 0.01
        return get_repetitions_from_exercise(metadata=metadata, frame_rate=frame_rate, kernel_size=kernel_size,
                                             threshold_multiplier=threshold_multiplier, threshold_peak=threshold_peak)
    else:
        return None


def get_exercise(metadata):
    exercise_counts = metadata['exercise'].value_counts()

    # Get the exercise with the highest count
    most_common_exercise = exercise_counts.idxmax()

    return most_common_exercise


def caloricOutput(exercise, total_duration, load_weight_lb, excercise_intensity, body_weight_lb):
    out = 0

    if exercise == "Squat":
        out = ((body_weight_lb + load_weight_lb) / 2.2) * ((total_duration / 60) / 60) * (
                5.5 + (excercise_intensity / 10) * 2.5)

    elif exercise == "Deadlift":
        out = ((body_weight_lb + load_weight_lb) / 2.2) * ((total_duration / 60) / 60) * (
                3.5 + (excercise_intensity / 10) * 2.5)

    elif exercise == "Pushup":
        out = ((body_weight_lb + load_weight_lb) / 2.2) * ((total_duration / 60) / 60) * (
                3.8 + (excercise_intensity / 10) * 4.2)

    return out
