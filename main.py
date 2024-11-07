import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from landmarks import (
    calculate_distance
)
from mp_hand_tracking import initialize_hand_detector, process_image
from plotting import plot

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    hands = initialize_hand_detector()
    plotter = plot()

    reference_scale = None
    scale = None
    estimated_distance = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        height = image.shape[0] 
        width = image.shape[1]

        results = process_image(hands, image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # world_landmarks = results.multi_hand_world_landmarks[0].landmark

            wrist = hand_landmarks.landmark[0]
            middle_finger = hand_landmarks.landmark[9]
            index_finger = hand_landmarks.landmark[5]
            pinky_finger = hand_landmarks.landmark[17]
            current_distance_vertical = calculate_distance(wrist, middle_finger, width, height)
            current_distance_horizontal = calculate_distance(index_finger, pinky_finger, width, height)

            if reference_scale == None:
                reference_scale = current_distance_vertical / current_distance_horizontal
                scale = reference_scale

            else:
                if current_distance_vertical/current_distance_horizontal >= reference_scale:
                    scale = 1 / current_distance_vertical
                
                else:
                    scale = 1 / reference_scale / current_distance_horizontal

            estimated_distance = scale * 200

            print (scale)


            center_x = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * width)
            center_y = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * height)

            parts = {
                "palm": [0, 5, 9, 13, 17, 0],
                "thumb": [0, 1, 2, 3, 4],
                "index": [5, 6, 7, 8],
                "middle": [9, 10, 11, 12],
                "ring": [13, 14, 15, 16],
                "pinky": [17, 18, 19, 20]
            }

            x_data = []
            y_data = []
            z_data = []


            for part in parts.values():
                x_part = []
                y_part = []
                z_part = []
                for idx in part:
                    landmark = hand_landmarks.landmark[idx]

                    scaled_x = (center_x / 200 + (landmark.x * width - center_x) * scale) / 5
                    scaled_y = (center_y / 200 + (landmark.y * height - center_y) * scale) / 5
                    scaled_z = estimated_distance / 2 + landmark.z

                    x_part.append(scaled_x)
                    y_part.append(scaled_y)
                    z_part.append(scaled_z)

                x_data.append(x_part)
                y_data.append(y_part)
                z_data.append(z_part)


            # print(scaled_x, scaled_y, scaled_z)

            plotter.update_scatter_plot(x_data, y_data, z_data)

            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        image = cv2.flip(image, 1)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
