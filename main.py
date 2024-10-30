import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from time import perf_counter
from dataclasses import dataclass
import torch
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float


def weightedSum(data, depth=10):
    weighted_sum = 0
    denom = 2 ** depth

    counter = 1

    for i in data[-depth:]:
        weighted_sum += (2 ** counter) * i / denom
        counter += 1

    return weighted_sum


def getAreaFromLandmarks(landmarks):

    min_x, max_x = landmarks[0].x, landmarks[0].x
    min_y, max_y = landmarks[0].y, landmarks[0].y

    marks = [1, 5, 9, 13, 17]

    for landmark in [landmarks[j] for j in marks]:

        if (landmark.x > max_x):
            max_x = landmark.x
        if (landmark.x < min_x):
            min_x = landmark.x
        if (landmark.y > max_y):
            max_y = landmark.y
        if (landmark.y < min_y):
            min_y = landmark.y

    return (min_x, min_y), (max_x, max_y), (max_x - min_x) * (max_y - min_y)


def transformScreenLandmarks(landmarks, image):
    return [FakeLandmark(landmark.x * image.shape[1], landmark.y * image.shape[0], 0) for landmark in landmarks]


def getAmountHandClosed(landmarks):
    vec1 = (landmarks[5].x - landmarks[0].x, landmarks[5].y -
            landmarks[0].y, landmarks[5].z - landmarks[0].z)
    vec2 = (landmarks[6].x - landmarks[5].x, landmarks[6].y -
            landmarks[5].y, landmarks[6].z - landmarks[5].z)

    return abs((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)))


# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        max_num_hands=1,
        min_tracking_confidence=0.5) as hands:

    plt.ion()

    line_screen, = plt.plot([], [], label="screen")
    line_world, = plt.plot([], [], label="world")
    line_div, = plt.plot([], [], label="combined")
    line_smoothed, = plt.plot([], [], label="combined_smoothed")

    plt.legend(loc="upper left")

    data_screen = [0]
    data_world = [0]
    data_div = [0]
    data_smoothed = [0]
    x_increment = [0]

    recording_data = False
    training_run = 0
    training_offset = 0
    start_z = None
    lock_z = None
    headers = ["time", "trainingRun", "expectedZ"] + \
        [f"world_landmark_{i}_{j}" for i in range(21) for j in [
            "x", "y", "z"]]

    with open("training_data.csv", "a") as f:
        f.write(",".join(headers) + "\n")

        while cap.isOpened():

            success, image = cap.read()

            if not success:
                continue

            # To improve performanceb, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            min_p = (0, 0)
            max_p = (0, 0)
            smin_p = (0, 0)
            smax_p = (0, 0)

            if results.multi_hand_landmarks:

                min_p, max_p, world_area = getAreaFromLandmarks(
                    results.multi_hand_world_landmarks[0].landmark)
                smin_p, smax_p, screen_area = getAreaFromLandmarks(
                    transformScreenLandmarks(results.multi_hand_landmarks[0].landmark, image))

                data_screen.append(screen_area / 100000)
                data_world.append(world_area * 100)

                # World area / screen area ^ 3/4
                z_loc = (world_area / (screen_area ** 0.5)) * 1e6

                alpha = 10 / (20 + 1)  # alpha ≈ 0.095
                data_smoothed.append(alpha * data_div[-1] + (1 - alpha) * (
                    data_smoothed[-1] if data_smoothed else data_div[-1]))

                z_loc_smooth = data_smoothed[-1]

                if start_z == None:
                    data_div.append(z_loc - training_offset)
                else:
                    data_div.append(data_div[-1])

                key = cv2.waitKey(10)
                if key == ord("r"):
                    recording_data = not recording_data
                    if recording_data:
                        training_run += 1
                if key == ord("c"):
                    if start_z == None:
                        start_z = z_loc
                    else:
                        training_offset = z_loc - start_z
                        start_z = None

                if key == ord("f"):
                    if lock_z == None:
                        lock_z = z_loc
                        recording_data = True
                    else:
                        lock_z = None
                        recording_data = False

                # # Weighted sum of last 20 datapoints in attempt to smooth out data
                # data_smoothed.append(weightedSum(data_div, 20))

                # data_smoothed.append(data_div[-1])

                x_increment.append(x_increment[-1] + 1)

                # line_screen.set_xdata(x_increment);
                # line_world.set_xdata(x_increment);
                line_div.set_xdata(x_increment)
                line_smoothed.set_xdata(x_increment)

                # line_screen.set_ydata(data_screen);
                # line_world.set_ydata(data_world);
                line_div.set_ydata(data_div)
                line_smoothed.set_ydata(data_smoothed)

                # , data_screen[-100:], data_world[-100:]))]) #max(max(data_div[-100:]), max(data_screen[-100:]), max(data_world[-100:]))])
                plt.axis([max(0, x_increment[-1] - 100), x_increment[-1],
                          0, max(map(max, data_smoothed[-100:], data_div[-100:]))])
                plt.draw()
                plt.pause(0.001)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            image = cv2.rectangle(image, (int(smax_p[0] * image.shape[1]), int(smax_p[1] * image.shape[0])), (int(
                smin_p[0] * image.shape[1]), int(smin_p[1] * image.shape[0])), (0, 255, 0), 3)
            image = cv2.rectangle(image, (int(max_p[0] * image.shape[1]), int(max_p[1] * image.shape[0])), (int(
                min_p[0] * image.shape[1]), int(min_p[1] * image.shape[0])), (255, 0, 0), 3)
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

            image = cv2.putText(image, f"Recording {recording_data}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording_data else (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, f"Training run {training_run}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            image = cv2.putText(image, f"Training offset {training_offset}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            image = cv2.putText(image, f"Z {z_loc}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            image = cv2.putText(image, f"Calibrating {start_z != None}",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if start_z != None else (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, f"Locked Z {lock_z}",
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if lock_z != None else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            if recording_data:
                f.write(f"{perf_counter()},{training_run},{z_loc if lock_z == None else lock_z}," + ",".join(
                    [str(landmark.x) + "," + str(landmark.y) + "," + str(landmark.z) for landmark in results.multi_hand_world_landmarks[0].landmark]) + "\n")

cap.release()
