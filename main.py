import cv2
import numpy as np
import math
from landmarks import calculate_distance, calculate_world_coordinates
from mp_hand_tracking import initialize_hand_detector, process_image, hand_connections
from plotting import plot
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from dataclasses import dataclass
import imutils

@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float

def getZ(landmark, offset):
  return landmark.z + offset;

def getAreaFromLandmarks(landmarks, a, b):

  return math.sqrt((landmarks[a].x - landmarks[b].x)**2 + (landmarks[a].y - landmarks[b].y)**2);
 
def transformScreenLandmarks(landmarks, image):
  return [FakeLandmark(landmark.x * image.shape[1], landmark.y * image.shape[0], 0) for landmark in landmarks];

def getAmountHandClosed(landmarks):
  vec1 = (landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z);
  vec2 = (landmarks[6].x - landmarks[5].x, landmarks[6].y - landmarks[5].y, landmarks[6].z - landmarks[5].z);

  return abs((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)));





def main():
    vs = WebcamVideoStream(src=0).start()   
    hands = initialize_hand_detector()
    plotter = plot()
    open_camera = True

    frame_counter = 0
    reset = True
    scale = None
    estimated_distance = None

    data_div = [];

    while open_camera:
        image = vs.read() 
        image = imutils.resize(image, width=1280)

        height = image.shape[0] 
        width = image.shape[1]

        results = process_image(hands, image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist_screen = hand_landmarks.landmark[0]
            middle_finger_screen = hand_landmarks.landmark[9]
            index_finger_screen = hand_landmarks.landmark[5]
            pinky_finger_screen = hand_landmarks.landmark[17]
            current_distance_vertical = calculate_distance(wrist_screen, middle_finger_screen, width, height)
            current_distance_horizontal = calculate_distance(index_finger_screen, pinky_finger_screen, width, height)

            world_landmarks = results.multi_hand_world_landmarks[0]
            centre_world = world_landmarks.landmark[0]

            # Initialize reference distances and scale factor
            if reset:
                reference_distance_vertical = current_distance_vertical
                reference_distance_horizontal = current_distance_horizontal
                reference_distance_depth = 1
                estimated_distance = reference_distance_depth
                reset = False
            
            scale_vertical = current_distance_vertical/reference_distance_vertical
            scale_horizontal = current_distance_horizontal/reference_distance_horizontal
            if (scale_vertical * 0.95 > scale_horizontal):
                estimated_distance = reference_distance_depth/scale_vertical
           
            else :
                estimated_distance = reference_distance_depth/scale_horizontal - (index_finger_screen.z + pinky_finger_screen.z)/2
            

            world_area = getAreaFromLandmarks(results.multi_hand_world_landmarks[0].landmark, 5, 17) + getAreaFromLandmarks(results.multi_hand_world_landmarks[0].landmark, 0, 9);
            screen_area = getAreaFromLandmarks(transformScreenLandmarks(results.multi_hand_landmarks[0].landmark, image), 5, 17) + getAreaFromLandmarks(transformScreenLandmarks(results.multi_hand_landmarks[0].landmark, image), 0, 9);

            # World area / screen area ^ 3/4
            data_div.append(world_area * 1000 / screen_area);

            # data_div[-1] -= data_div[-1] * 0.3 * ((1 - getAmountHandClosed(results.multi_hand_world_landmarks[0].landmark)) ** 2)# - 5 ** (1 - getAmountHandTilted(results.multi_hand_world_landmarks[0].landmark)));
            data_div[-1] = data_div[-1] * 3 - 0.6

            if (len(data_div) >= 2):
                data_div[-1] = data_div[-2] * 0.6 + data_div[-1] * 0.4

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
            z_test = []

            # print(hand_landmarks.landmark[0].y, wrist_screen.y, calculate_world_coordinates(centre_world, world_landmarks.landmark[5]).y + wrist_screen.y)


            for part in parts.values():
                x_part = []
                y_part = []
                z_part = []
                for idx in part:
                    landmark = calculate_world_coordinates(centre_world, world_landmarks.landmark[idx])

                    scaled_x = landmark.x + wrist_screen.x
                    scaled_y = landmark.y + wrist_screen.y
                    scaled_z = landmark.z + estimated_distance

                    x_part.append(scaled_x)
                    y_part.append(scaled_y)
                    z_part.append(scaled_z)

                    
                z_test.append(list(map(lambda landmark : getZ(landmark, data_div[-1] / 5), [results.multi_hand_world_landmarks[0].landmark[start] for start in part])))
                x_data.append(x_part)
                y_data.append(y_part)
                z_data.append(z_part)


           
            plotter.update_scatter_plot(x_data, y_data, z_test)
            # plotter.update_2d_plot(frame_counter, x_data[0][0], y_data[0][0], z_data[0][0])
            frame_counter += 1

            hand_connections(image, hand_landmarks, results.multi_hand_landmarks)

        image = cv2.flip(image, 1)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    vs.stop().release()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
