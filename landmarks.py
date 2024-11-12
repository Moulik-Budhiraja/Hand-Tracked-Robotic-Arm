####################
#
# file: landmarks.py
#
# description: Contains dataclasses and functions related to transforming MediaPipe landmarks.
#
####################

from dataclasses import dataclass
import numpy as np
import math

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

def get_length_from_landmarks(landmarks, a, b):
    return math.sqrt((landmarks[a].x - landmarks[b].x)**2 + (landmarks[a].y - landmarks[b].y)**2);
 
def transform_screen_landmarks(landmarks, image):
    return [Coordinate(landmark.x * image.shape[1], landmark.y * image.shape[0], 0) for landmark in landmarks];

def get_amount_hand_closed(landmarks):
    vec1 = (landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z);
    vec2 = (landmarks[6].x - landmarks[5].x, landmarks[6].y - landmarks[5].y, landmarks[6].z - landmarks[5].z);

    return abs((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)));

def get_amount_hand_tilted(landmarks):
    vec = (landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z);
    return max(0, -vec[2] / math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2))

def calculate_distance(landmark1, landmark2, width, height):
    dx = (landmark1.x - landmark2.x) * width
    dy = (landmark1.y - landmark2.y) * height
    return np.sqrt(dx*dx + dy*dy)

def calculate_world_coordinates(centre, landmark):
    world_coords = Coordinate
    world_coords.x = landmark.x - centre.x 
    world_coords.y = landmark.y - centre.y
    world_coords.z = landmark.z - centre.z
    return world_coords

def calculate_center(point, center, sensitivity):
    center.x = ((point.x - center.x) * (1 - sensitivity)) + center.x
    center.y = ((point.y - center.y) * (1 - sensitivity)) + center.y
    center.z = ((point.z - center.z) * (1 - sensitivity)) + center.z

    return center