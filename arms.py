import math
from dataclasses import dataclass
import serial


@dataclass
class Coordinate:
    x: float
    y: float
    z: float


class Arms:
    def __init__(self):
        self.first_arm_length = 0.478
        self.second_arm_length = 0.522

        self.sends = 0

    def base_rotation(self, coordinate):
        vector2 = [coordinate.x, coordinate.y]
        norm2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if norm2 == 0:
            return

        degrees = math.acos(vector2[1] / norm2)
        if coordinate.x >= 0:
            return degrees
        else:
            return degrees * -1

    def check_in_range(self, coordinate):
        distance = math.sqrt(coordinate.x**2 + coordinate.y**2 + coordinate.z**2)
        if distance > 1 or distance <= 0:
            return False
        else:
            return True

    def find_angles(self, coordinate):
        distance = math.sqrt(coordinate.x**2 + coordinate.y**2 + coordinate.z**2)
        vector_angle = math.asin(coordinate.z / distance)
        alpha = math.acos((distance**2 + self.first_arm_length**2 - self.second_arm_length**2) / (2 * distance * self.first_arm_length))
        first_angle = vector_angle + alpha
        beta = math.acos(self.first_arm_length / self.second_arm_length * math.sqrt(1 - (math.cos(alpha) ** 2)))
        second_angle = math.pi / 2 + alpha - beta
        base_angle = self.base_rotation(coordinate)

        with serial.Serial("COM3", 115200, timeout=100, dsrdtr=None) as ser:

            ser.setRTS(False)
            ser.setDTR(False)

            data1 = bytes([0, min(255, max(0, int(first_angle * 255 / math.pi)))])
            data2 = bytes([1, min(255, max(0, 255 - int((math.pi / 2 - second_angle) * 255 / math.pi)))])

            ser.write(data1)
            ser.write(data2)

        return [base_angle, first_angle, second_angle]

    def find_point(self, coordinate):
        angles = self.find_angles(coordinate)
        alpha = angles[0]
        beta = angles[1]
        gamma = angles[2]

        point = Coordinate(
            x=self.first_arm_length * math.sin(alpha) * math.cos(beta),
            y=self.first_arm_length * math.cos(alpha) * math.cos(beta),
            z=self.first_arm_length * math.sin(beta),
        )
        len = math.sqrt((point.x - coordinate.x) ** 2 + (point.y - coordinate.y) ** 2 + (point.z - coordinate.z) ** 2)
        # print(len)
        return point
