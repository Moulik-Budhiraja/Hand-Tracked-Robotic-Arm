####################
#
# file: mp_hand_tracking.py
#
# description: Contains functions related to MediaPipe actions including initialization of the hand detector, 
#              processing the image, an drawing landmarks on the screen.
#
####################

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def initialize_hand_detector(model_complexity=1, min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5):
    hands = mp_hands.Hands(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        max_num_hands=max_num_hands,
        min_tracking_confidence=min_tracking_confidence
    )
    return hands

def process_image(hands, image):
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    image.flags.writeable = True
    return results

def hand_connections(image, landmarks):
    for hand_landmarks in landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(circle_radius=0, thickness=0),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )