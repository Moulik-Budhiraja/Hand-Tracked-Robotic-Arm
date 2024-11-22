####################
#
# file: main.py
#
# description: Converts real-time video footage of hand movement into 3d coordinates for the hand.
#
####################

import cv2
from landmarks import (
    Coordinate,
    get_amount_hand_tilted,
    transform_screen_landmarks,
    get_length_from_landmarks,
    calculate_world_coordinates,
    calculate_center,
)
from mp_hand_tracking import initialize_hand_detector, process_image, hand_connections
from plotting import plotting_process
from imutils.video import WebcamVideoStream
from arms import Arms
import imutils
import multiprocessing


def main():

    # Starting multiprocess
    plot_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()

    plot_proc = multiprocessing.Process(target=plotting_process, args=(plot_queue, control_queue))
    plot_proc.start()

    # Initialize webcam and hand tracking
    vs = WebcamVideoStream(src=0).start()
    hands = initialize_hand_detector()
    open_camera = True

    frame_counter = 0
    center = Coordinate(x=0, y=0.5, z=0.25)
    wrist_location = Coordinate(x=0, y=0, z=0)
    sensitivity = 1
    prev_sensitivity = 1

    arms = Arms()

    while open_camera:
        key = cv2.waitKey(20)
        image = vs.read()
        image = imutils.resize(image, width=400, height=300)

        results = process_image(hands, image)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]
            wrist_screen = hand_landmarks.landmark[0]
            world_landmarks = results.multi_hand_world_landmarks[0]
            centre_world = world_landmarks.landmark[0]

            # Calculating depth (wrist z coordinate)
            world_length = get_length_from_landmarks(results.multi_hand_world_landmarks[0].landmark, 5, 17) + get_length_from_landmarks(
                results.multi_hand_world_landmarks[0].landmark, 0, 9
            )
            screen_length = get_length_from_landmarks(
                transform_screen_landmarks(results.multi_hand_landmarks[0].landmark, image), 5, 17
            ) + get_length_from_landmarks(transform_screen_landmarks(results.multi_hand_landmarks[0].landmark, image), 0, 9)

            wrist_screen.z = world_length * 200 / screen_length
            wrist_screen.z += wrist_screen.z * 0.65 * get_amount_hand_tilted(results.multi_hand_world_landmarks[0].landmark) ** 2.3 + 0.05

            parts = {
                "palm": [0, 5, 9, 13, 17, 0],
                "thumb": [0, 1, 2, 3, 4],
                "index": [5, 6, 7, 8],
                "middle": [9, 10, 11, 12],
                "ring": [13, 14, 15, 16],
                "pinky": [17, 18, 19, 20],
            }

            x_data = []
            y_data = []
            z_data = []

            # Creating coordinate datapoints for hand
            for part in parts.values():
                x_part = []
                y_part = []
                z_part = []
                for idx in part:
                    landmark = calculate_world_coordinates(centre_world, world_landmarks.landmark[idx])

                    # wrist_screen.x = wrist_screen * sensitivity + center.x
                    # wrist_screen.y = wrist_screen * sensitivity + center.y

                    wrist_depth = ((wrist_screen.z - 0.5) * sensitivity * 2) + center.z
                    if wrist_depth <= -0.5:
                        wrist_depth = -0.5

                    if wrist_depth >= 1:
                        wrist_depth = 1

                    scaled_x = -1 * ((landmark.x) + ((wrist_screen.x - 0.5) * 4 / 3 * sensitivity) - center.x)
                    scaled_y = -1 * ((landmark.y) + ((wrist_screen.y - 0.5) * sensitivity) - center.y)
                    scaled_z = (landmark.z) + wrist_depth

                    if idx == 0:
                        wrist_location.x = scaled_x
                        wrist_location.y = scaled_y
                        wrist_location.z = scaled_z

                    x_part.append(scaled_x)
                    y_part.append(scaled_y)
                    z_part.append(scaled_z)

                x_data.append(x_part)
                y_data.append(y_part)
                z_data.append(z_part)

            if arms.check_in_range(wrist_location):
                arm_point = arms.find_point(wrist_location)
            else:
                arm_point = None

            plot_queue.put(
                {
                    "type": "scatter",
                    "x_data": x_data,
                    "y_data": y_data,
                    "z_data": z_data,
                    "center": center,
                    "sensitivity": sensitivity,
                    "arm_point": arm_point,
                    "wrist_point": wrist_location,
                }
            )

            frame_counter += 1

            hand_connections(image, results.multi_hand_landmarks)

        image = cv2.flip(image, 1)

        cv2.imshow("MediaPipe Hands", image)

        # Handle key presses
        if key & 0xFF == ord("a"):
            if sensitivity > 0.1:
                print("decreased sensitivity: ", sensitivity)
                sensitivity -= 0.1
            else:
                print("min sensitivity reached")
        if key & 0xFF == ord("d"):
            if sensitivity < 1:
                print("increased sensitivity: ", sensitivity)
                sensitivity += 0.1
            else:
                print("max sensitivity reached")
        if key & 0xFF == ord("c"):
            reset = True
            print("calibrated")
        if key & 0xFF == ord("q"):
            open_camera = False

        # print(wrist_location.x)
        # Update center based on sensitivity changes
        if sensitivity < prev_sensitivity:
            # print(wrist_location, center)
            center = calculate_center(wrist_location, center, sensitivity / prev_sensitivity)
            prev_sensitivity = sensitivity
        if sensitivity > prev_sensitivity:
            origin = Coordinate(x=0, y=0.5, z=0.25)
            center = calculate_center(center, origin, sensitivity)
            prev_sensitivity = sensitivity

    # Send termination signal to plotting process
    plot_queue.put("TERMINATE")
    plot_proc.join()

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
