# utils.py

import math
from dataclasses import dataclass
from typing import List
import torch
from model import model


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float


@dataclass
class PlotData:
    screen_area: float
    world_area: float
    data_div_value: float
    x_palm_data: List[float]
    y_palm_data: List[float]
    z_palm_data: List[float]
    x_thumb_data: List[float]
    y_thumb_data: List[float]
    z_thumb_data: List[float]
    x_index_data: List[float]
    y_index_data: List[float]
    z_index_data: List[float]
    x_middle_data: List[float]
    y_middle_data: List[float]
    z_middle_data: List[float]
    x_ring_data: List[float]
    y_ring_data: List[float]
    z_ring_data: List[float]
    x_pinky_data: List[float]
    y_pinky_data: List[float]
    z_pinky_data: List[float]


def weightedSum(data, depth=10):
    weighted_sum = 0
    denom = 2**depth
    counter = 1
    for i in data[-depth:]:
        weighted_sum += (2**counter) * i / denom
        counter += 1
    return weighted_sum


def getAreaFromLandmarks(landmarks):
    min_x, max_x = landmarks[0].x, landmarks[0].x
    min_y, max_y = landmarks[0].y, landmarks[0].y
    marks = [0, 5, 9, 13, 17]
    for landmark in [landmarks[j] for j in marks]:
        if landmark.x > max_x:
            max_x = landmark.x
        if landmark.x < min_x:
            min_x = landmark.x
        if landmark.y > max_y:
            max_y = landmark.y
        if landmark.y < min_y:
            min_y = landmark.y
    return (min_x, min_y), (max_x, max_y), (max_x - min_x) * (max_y - min_y)


def transformScreenLandmarks(landmarks, image):
    return [FakeLandmark(landmark.x * image.shape[1], landmark.y * image.shape[0], 0) for landmark in landmarks]


def getAmountHandClosed(landmarks):
    vec1 = (landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z)
    vec2 = (landmarks[6].x - landmarks[5].x, landmarks[6].y - landmarks[5].y, landmarks[6].z - landmarks[5].z)
    return abs(
        (vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
        / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2))
    )


def getAmountHandTilted(landmarks):
    up_vec = (0, -1, 0)
    hand_vec = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y, landmarks[9].z - landmarks[0].z)
    return (up_vec[0] * hand_vec[0] + up_vec[1] * hand_vec[1] + up_vec[2] * hand_vec[2]) / (
        math.sqrt(up_vec[0] ** 2 + up_vec[1] ** 2 + up_vec[2] ** 2) * math.sqrt(hand_vec[0] ** 2 + hand_vec[1] ** 2 + hand_vec[2] ** 2)
    )


def getX(landmark):
    return -1 * landmark.x


def getY(landmark):
    return -landmark.y


def getZ(landmark, offset):
    return landmark.z + offset


def getSaveX(landmarks, i):
    return landmarks[i].x - landmarks[0].x


def getSaveY(landmarks, i):
    return landmarks[i].y - landmarks[0].y


def getSaveZ(landmarks, i):
    return landmarks[i].z - landmarks[0].z


def getZOffset(landmarks):
    transformed = [getFunc(landmarks, i) for i in range(21) for getFunc in [getSaveX, getSaveY, getSaveZ]]

    with torch.no_grad():
        return model(torch.tensor([transformed], dtype=torch.float32).unsqueeze(0)).item()
