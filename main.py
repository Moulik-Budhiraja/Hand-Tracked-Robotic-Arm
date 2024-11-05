# main.py

import cv2
import mediapipe as mp
import time
from multiprocessing import Process, Queue
from utils import (
    weightedSum,
    getAreaFromLandmarks,
    transformScreenLandmarks,
    getAmountHandClosed,
    getAmountHandTilted,
    getX,
    getY,
    getZ,
    getSaveX,
    getSaveY,
    getSaveZ,
    getZOffset,
    PlotData,
)
import queue


def main():
    base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task", delegate=mp.tasks.BaseOptions.Delegate.GPU)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize multiprocessing Queue with limited size
    data_queue = Queue(maxsize=10)

    # Start plotting process
    from plotting import plotting_process

    plot_process = Process(target=plotting_process, args=(data_queue,))
    plot_process.start()

    recording = False
    start_z = None

    with open("data.csv", "w") as f:
        f.write("time,est_z," + ",".join(f"lmk_{i}_{j}" for i in range(21) for j in ["x", "y", "z"]) + "\n")
        with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                # To improve performance, mark the image as not writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                min_p = (0, 0)
                max_p = (0, 0)
                smin_p = (0, 0)
                smax_p = (0, 0)

                if results.multi_hand_landmarks:
                    # Get area metrics
                    min_p, max_p, world_area = getAreaFromLandmarks(results.multi_hand_world_landmarks[0].landmark)
                    smin_p, smax_p, screen_area = getAreaFromLandmarks(transformScreenLandmarks(results.multi_hand_landmarks[0].landmark, image))

                    # Scale areas
                    screen_area /= 100000
                    world_area *= 100

                    # Calculate data_div_value
                    data_div_value = world_area / (screen_area**0.5)
                    # data_div_value -= data_div_value * 0.3 * ((1 - getAmountHandClosed(results.multi_hand_world_landmarks[0].landmark)) ** 2)

                    z_off = getZOffset(results.multi_hand_landmarks[0].landmark)
                    data_div_value -= z_off
                    data_div_value = data_div_value - 1

                    key = cv2.waitKey(5) & 0xFF
                    if key == ord("c"):
                        recording = not recording
                        if recording:
                            start_z = data_div_value
                        else:
                            start_z = None

                    if recording:
                        f.write(
                            f"{time.time()},"
                            + f"{data_div_value - start_z},"
                            + ",".join(
                                f"{getFunc(results.multi_hand_landmarks[0].landmark, i)}"
                                for i in range(21)
                                for getFunc in [getSaveX, getSaveY, getSaveZ]
                            )
                            + "\n"
                        )

                    # Define the lines for different parts of the hand
                    lines = [
                        [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)],  # Palm
                        [(0, 1), (1, 2), (2, 3), (3, 4)],  # Thumb
                        [(5, 6), (6, 7), (7, 8)],  # Index
                        [(9, 10), (10, 11), (11, 12)],  # Middle
                        [(13, 14), (14, 15), (15, 16)],  # Ring
                        [(17, 18), (18, 19), (19, 20)],  # Pinky
                    ]

                    # Extract data for each part
                    # Palm
                    x_palm_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[0]]
                    y_palm_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[0]]
                    z_palm_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[0]]
                    x_palm_data.append(x_palm_data[0])
                    y_palm_data.append(y_palm_data[0])
                    z_palm_data.append(z_palm_data[0])

                    # Thumb
                    x_thumb_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[1]]
                    y_thumb_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[1]]
                    z_thumb_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[1]]

                    # Index
                    x_index_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[2]]
                    y_index_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[2]]
                    z_index_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[2]]

                    # Middle
                    x_middle_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[3]]
                    y_middle_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[3]]
                    z_middle_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[3]]

                    # Ring
                    x_ring_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[4]]
                    y_ring_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[4]]
                    z_ring_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[4]]

                    # Pinky
                    x_pinky_data = [getZ(results.multi_hand_world_landmarks[0].landmark[start], data_div_value / 5) for start, end in lines[5]]
                    y_pinky_data = [getX(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[5]]
                    z_pinky_data = [getY(results.multi_hand_world_landmarks[0].landmark[start]) for start, end in lines[5]]

                    # Prepare the data object to send to the plotting process
                    plot_data = PlotData(
                        screen_area=screen_area,
                        world_area=world_area,
                        data_div_value=data_div_value,
                        x_palm_data=x_palm_data,
                        y_palm_data=y_palm_data,
                        z_palm_data=z_palm_data,
                        x_thumb_data=x_thumb_data,
                        y_thumb_data=y_thumb_data,
                        z_thumb_data=z_thumb_data,
                        x_index_data=x_index_data,
                        y_index_data=y_index_data,
                        z_index_data=z_index_data,
                        x_middle_data=x_middle_data,
                        y_middle_data=y_middle_data,
                        z_middle_data=z_middle_data,
                        x_ring_data=x_ring_data,
                        y_ring_data=y_ring_data,
                        z_ring_data=z_ring_data,
                        x_pinky_data=x_pinky_data,
                        y_pinky_data=y_pinky_data,
                        z_pinky_data=z_pinky_data,
                    )

                    # Send data to the plotting process
                    try:
                        data_queue.put_nowait(plot_data)
                    except queue.Full:
                        # If the queue is full, discard the oldest data and enqueue the new one
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait(plot_data)
                        except queue.Empty:
                            pass  # If the queue was empty after removing, just pass

                    # Draw hand landmarks on the image
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                        # Draw bounding rectangles
                        # image = cv2.rectangle(image, (int(smin_p[0]), int(smin_p[1])), (int(smax_p[0]), int(smax_p[1])), (0, 255, 0), 3)
                        # image = cv2.rectangle(
                        #     image,
                        #     (int(smin_p[0]), int(smin_p[1])),
                        #     (
                        #         int(smax_p[0]),
                        #         int(
                        #             smin_p[1]
                        #             + (max_p[1] * image.shape[0] - min_p[1] * image.shape[0])
                        #             * (smax_p[0] - smin_p[0])
                        #             / (max_p[0] * image.shape[1] - min_p[0] * image.shape[1])
                        #         ),
                        #     ),
                        #     (255, 0, 0),
                        #     3,
                        # )
                        # image = cv2.rectangle(
                        #     image,
                        #     (int(smin_p[0]), int(smin_p[1])),
                        #     (
                        #         int(smin_p[0] + (max_p[0] * image.shape[1] - min_p[0] * image.shape[1])),
                        #         int(smin_p[1] + (max_p[1] * image.shape[0] - min_p[1] * image.shape[0])),
                        #     ),
                        #     (255, 0, 0),
                        #     3,
                        # )
                # Flip the image horizontally for a later selfie-view display
                image = cv2.flip(image, 1)

                image = cv2.putText(
                    image, f"Recording: {recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2
                )

                # Display the image
                cv2.imshow("MediaPipe Hands", cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2)))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    # Terminate plotting process
    data_queue.put(None)
    plot_process.join()
    cap.release()


if __name__ == "__main__":
    main()
