import cv2
import numpy as np

from landmarks import calculate_distance, calculate_world_coordinates
from mp_hand_tracking import initialize_hand_detector, process_image, hand_connections
from plotting import plot
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils



def main():
    vs = WebcamVideoStream(src=0).start()   
    hands = initialize_hand_detector()
    plotter = plot()
    open_camera = True

    frame_counter = 0
    reset = True
    estimated_distance = None

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
                    landmark = calculate_world_coordinates(centre_world, world_landmarks.landmark[idx])

                    scaled_x = landmark.x + wrist_screen.x
                    scaled_y = landmark.y + wrist_screen.y
                    scaled_z = landmark.z + estimated_distance

                    x_part.append(scaled_x)
                    y_part.append(scaled_y)
                    z_part.append(scaled_z)

                x_data.append(x_part)
                y_data.append(y_part)
                z_data.append(z_part)



            plotter.update_scatter_plot(x_data, y_data, z_data)
            plotter.update_2d_plot(frame_counter, x_data[0][0], y_data[0][0], z_data[0][0])
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
