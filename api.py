from inference import get_exercise, get_repetitions, \
    generate_prediction_metadata_from_video, caloricOutput

# This file is used for testing the response generation only, user input is static (not received from http request)
# make columns list
columns = ['box_width', 'box_height']
for i in range(17):
    columns.append(f'kp_{i + 1}_x')
    columns.append(f'kp_{i + 1}_y')

# user inputted values received from http POST request
intensity = 5
body_weight = 190
load = 135
# video path of saved file from http request
video_path = "/Users/andycraig/PycharmProjects/pythonProject/bilbo_swaggins181/2024-01-06_22-04-20_UTC_1.mp4"

# debug purposes
metadata_path = "temp.csv"
# original frame rate of video
frame_rate_original = 30
# frame rate for analysis purposes
frame_rate_processed = 10


metadata = generate_prediction_metadata_from_video(video_path=video_path,
                                                   skip_frames=int(frame_rate_original / frame_rate_processed),
                                                   columns=columns)
# get the most common exercise
exercise = get_exercise(metadata)
# get repetition data
squat_height_data, active_duration, avg_rep_time, reps = get_repetitions(exercise=exercise, metadata=metadata, frame_rate=frame_rate_processed)
# calculate total duration of video
total_duration = len(metadata) / frame_rate_processed
# calculate calories burned using MET equations
calories_burned = caloricOutput(exercise=exercise, load_weight_lb=load,total_duration=total_duration, excercise_intensity=intensity, body_weight_lb=body_weight)
# create response dict
response = {
        'repetitions': reps,
        'total_duration': total_duration,
        'active_duration': active_duration,
        'calories_burned': calories_burned,
        'exercise': exercise,
        'intensity': intensity,
        'body_weight': body_weight,
        'load': load,
        'average_rep_time': avg_rep_time,
        'squat_height_data': list(squat_height_data)
    }

print(response)
