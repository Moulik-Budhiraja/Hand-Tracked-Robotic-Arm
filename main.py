# main.py

import cv2
from dataclasses import dataclass
from landmarks import calculate_distance, calculate_world_coordinates, calculate_center
from mp_hand_tracking import initialize_hand_detector, process_image, hand_connections
from plotting import plotting_process
from imutils.video import WebcamVideoStream
import imutils
import multiprocessing
import time

@dataclass
class coordinates:
    x: float
    y: float
    z: float

def main():
    plot_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()

    plot_proc = multiprocessing.Process(target=plotting_process, args=(plot_queue, control_queue))
    plot_proc.start()

    vs = WebcamVideoStream(src=0).start()   
    hands = initialize_hand_detector()
    open_camera = True

    frame_counter = 0
    reset = True
    estimated_distance = None
    prev_estimated_distance = None
    center = coordinates(x=0, y=0, z=0)
    sensitivity = 1
    prev_sensitivity = 1

    while open_camera:
        key = cv2.waitKey(20)
        image = vs.read() 
        image = imutils.resize(image, width=400)

        height = image.shape[0] 
        width = image.shape[1]

        results = process_image(hands, image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist_screen, middle_knuckle_screen = hand_landmarks.landmark[0], hand_landmarks.landmark[9]
            index_knuckle_screen, pinky_knuckle_screen = hand_landmarks.landmark[5], hand_landmarks.landmark[17]
            current_distance_vertical = calculate_distance(wrist_screen, middle_knuckle_screen, width, height)
            current_distance_horizontal = calculate_distance(index_knuckle_screen, pinky_knuckle_screen, width, height)

            world_landmarks = results.multi_hand_world_landmarks[0]
            centre_world = world_landmarks.landmark[0]

            if reset:
                reference_distance_vertical = current_distance_vertical
                reference_distance_horizontal = current_distance_horizontal
                reference_distance_depth = 1
                estimated_distance = reference_distance_depth
                prev_estimated_distance = reference_distance_depth
                reset = False
            else:
                prev_estimated_distance = estimated_distance

            scale_vertical = current_distance_vertical / reference_distance_vertical
            scale_horizontal = current_distance_horizontal / reference_distance_horizontal
            if (scale_vertical * 0.95 > scale_horizontal):
                estimated_distance = reference_distance_depth / scale_vertical
            else:
                estimated_distance = reference_distance_depth / scale_horizontal - (index_knuckle_screen.z + pinky_knuckle_screen.z) / 2 

            if reset:
                pass 
            else:
                estimated_distance = 0.4 * prev_estimated_distance + 0.6 * estimated_distance - 0.3

            wrist_screen.z = estimated_distance

            wrist_screen.x -= 0.5
            wrist_screen.y -= 0.5
            wrist_screen.z -= 0.5

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

                    scaled_x = (landmark.x) + (wrist_screen.x * sensitivity) + center.x
                    scaled_y = (landmark.y) + (wrist_screen.y * sensitivity) + center.y 
                    scaled_z = (landmark.z) + (wrist_screen.z * sensitivity) + center.z 

                    x_part.append(scaled_x)
                    y_part.append(scaled_y)
                    z_part.append(scaled_z)

                x_data.append(x_part)
                y_data.append(y_part)
                z_data.append(z_part)

            plot_queue.put({
                "type": "scatter",
                "x_data": x_data,
                "y_data": y_data,
                "z_data": z_data,
                "center": center,
                "sensitivity": sensitivity
            })

            # plot_queue.put({
            #     "type": "2d",
            #     "frame": frame_counter,
            #     "x": x_data[0][0],
            #     "y": y_data[0][0],
            #     "z": z_data[0][0]
            # })

            frame_counter += 1

            hand_connections(image, hand_landmarks, results.multi_hand_landmarks)

        image = cv2.flip(image, 1)

        cv2.imshow('MediaPipe Hands', image)

        # Handle key presses
        if key & 0xFF == ord('a'):
            if sensitivity > 0.1:
                print("decreased sensitivity")
                sensitivity -= 0.1
            else:
                print("min sensitivity reached")
        if key & 0xFF == ord('d'):
            if sensitivity < 1:
                print("increased sensitivity")
                sensitivity += 0.1
            else:
                print("max sensitivity reached")
        if key & 0xFF == ord('c'):
            reset = True
            print("calibrated")
        if key & 0xFF == ord('q'):
            open_camera = False

        # Update center based on sensitivity changes
        if sensitivity < prev_sensitivity:
            center = calculate_center(wrist_screen, center, sensitivity / prev_sensitivity)
            prev_sensitivity = sensitivity
        if sensitivity > prev_sensitivity:
            origin = coordinates(x=0, y=0, z=0)
            center = calculate_center(center, origin, sensitivity)
            prev_sensitivity = sensitivity

    # Send termination signal to plotting process
    plot_queue.put("TERMINATE")
    plot_proc.join()

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    main()
