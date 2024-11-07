# landmarks.py

from dataclasses import dataclass
import numpy as np
import math

# @dataclass
# class FakeLandmark:
#     x: float
#     y: float
#     z: float

def calculate_distance(landmark1, landmark2, width, height):
    dx = (landmark1.x - landmark2.x) * width
    dy = (landmark1.y - landmark2.y) * height
    return np.sqrt(dx*dx + dy*dy)


# def get_distance_from_landmarks(landmarks, a, b):
#     return math.sqrt((landmarks[a].x - landmarks[b].x) ** 2 + (landmarks[a].y - landmarks[b].y) ** 2)

# def get_amount_hand_closed(landmarks):
#     vec1 = (
#         landmarks[5].x - landmarks[0].x,
#         landmarks[5].y - landmarks[0].y,
#         landmarks[5].z - landmarks[0].z
#     )
#     vec2 = (
#         landmarks[6].x - landmarks[5].x,
#         landmarks[6].y - landmarks[5].y,
#         landmarks[6].z - landmarks[5].z
#     )

#     dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
#     magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)
#     magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)

#     if magnitude1 == 0 or magnitude2 == 0:
#         return 0

#     return abs(dot_product / (magnitude1 * magnitude2))

# def getX(landmark):
#     return -1 * landmark.x

# def getY(landmark):
#     return -1 * landmark.y

# def getZ(landmark):
#     return landmark.z