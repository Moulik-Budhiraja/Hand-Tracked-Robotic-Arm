import asyncio
import time
import numpy as np
import cv2
import threading

from winsdk.windows.media.capture import (
    MediaStreamType,
    MediaCapture,
    MediaCaptureInitializationSettings,
    MediaCaptureSharingMode,
    MediaCaptureMemoryPreference,
    StreamingCaptureMode
)
from winsdk.windows.media.capture.frames import (
    MediaFrameSourceGroup,
    MediaFrameSourceInfo,
    MediaFrameSourceKind,
    MediaFrameReader
)
from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
from winsdk.windows.storage.streams import Buffer

# Desired camera resolution
CAMERA_IMAGE_WIDTH = 640
CAMERA_IMAGE_HEIGHT = 360

# Global variables to store frames
infra_frame = None
color_frame = None
infra_lock = threading.Lock()
color_lock = threading.Lock()
exit_event = threading.Event()


async def capture_infrared():
    global infra_frame

    ##### Find the first Infrared source group #####
    media_frame_source_groups = await MediaFrameSourceGroup.find_all_async()

    infra_source_group = None
    infra_source_info = None

    # Iterate through source groups to find Infrared source
    for source_group in media_frame_source_groups:
        for source_info in source_group.source_infos:
            print(f"Stream Type: {source_info.media_stream_type}, Source Kind: {
                  source_info.source_kind}")
            if (source_info.media_stream_type == MediaStreamType.VIDEO_RECORD and
                    source_info.source_kind == MediaFrameSourceKind.INFRARED):
                infra_source_group = source_group
                infra_source_info = source_info
                break  # Found the desired source, exit inner loop
        if infra_source_group:
            break  # Exit outer loop if source is found

    # Check if Infrared source is available
    if infra_source_group is None or infra_source_info is None:
        print("No Infrared source was found!")
        return

    ##### Initialize MediaCapture for Infrared #####
    media_capture = MediaCapture()
    try:
        # Settings for MediaCapture object
        media_capture_settings = MediaCaptureInitializationSettings()
        media_capture_settings.source_group = infra_source_group
        media_capture_settings.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL
        media_capture_settings.memory_preference = MediaCaptureMemoryPreference.CPU
        media_capture_settings.streaming_capture_mode = StreamingCaptureMode.VIDEO

        await media_capture.initialize_async(media_capture_settings)

        # Initialize frame reader for Infrared
        frame_source = media_capture.frame_sources.get(infra_source_info.id)
        if frame_source is None:
            print("Failed to get Infrared frame source!")
            return

        media_frame_reader: MediaFrameReader = await media_capture.create_frame_reader_async(frame_source)
        await media_frame_reader.start_async()

        # Buffer for Infrared frame data
        frame_buffer = Buffer(CAMERA_IMAGE_WIDTH * CAMERA_IMAGE_HEIGHT * 4)

        print("Starting to capture Infrared frames...")

        while not exit_event.is_set():
            # Capture at approximately 20fps
            await asyncio.sleep(0.05)

            # Acquire the latest Infrared frame
            media_frame_reference = media_frame_reader.try_acquire_latest_frame()

            if media_frame_reference is None:
                continue

            # Process the Infrared frame
            with media_frame_reference:
                video_media_frame = media_frame_reference.video_media_frame

                if video_media_frame is None:
                    continue

                software_bitmap = video_media_frame.software_bitmap

                if software_bitmap is None:
                    continue

                with software_bitmap:
                    # Convert to RGBA8
                    infrared_rgba = SoftwareBitmap.convert(
                        software_bitmap, BitmapPixelFormat.RGBA8)
                    # Copy to buffer
                    infrared_rgba.copy_to_buffer(frame_buffer)
                    # Create OpenCV image
                    image_array = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(
                        CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH, 4)
                    # Convert from RGBA to BGR
                    infrared_bgr = cv2.cvtColor(
                        image_array, cv2.COLOR_RGBA2BGR)

                    # Update the global Infrared frame
                    with infra_lock:
                        infra_frame = infrared_bgr

    except Exception as e:
        print(f"An error occurred in Infrared capture: {e}")
    finally:
        # Stop the frame reader and clean up
        if 'media_frame_reader' in locals():
            await media_frame_reader.stop_async()
        media_capture.close()
        print("Infrared capture stopped.")


def capture_color():
    global color_frame

    # Initialize OpenCV VideoCapture for the Color camera
    cap = cv2.VideoCapture(0)  # Change the index if necessary

    # Set resolution if desired
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_IMAGE_HEIGHT)

    if not cap.isOpened():
        print("Cannot open Color camera")
        return

    print("Starting to capture Color frames...")

    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab Color frame")
            break

        # Update the global Color frame
        with color_lock:
            color_frame = frame

        # Sleep briefly to reduce CPU usage
        time.sleep(0.05)  # Approx 20fps

    # Release the capture when done
    cap.release()
    print("Color capture stopped.")


async def display_frames():
    while not exit_event.is_set():
        # Retrieve the latest Infrared and Color frames
        with infra_lock:
            infra = infra_frame.copy() if infra_frame is not None else None
        with color_lock:
            color = color_frame.copy() if color_frame is not None else None

        if infra is not None and color is not None:
            # Resize if necessary to match heights
            if infra.shape[0] != color.shape[0]:
                color = cv2.resize(
                    color, (int(color.shape[1] * infra.shape[0] / color.shape[0]), infra.shape[0]))
            elif color.shape[0] != infra.shape[0]:
                infra = cv2.resize(
                    infra, (int(infra.shape[1] * color.shape[0] / infra.shape[0]), color.shape[0]))

            # Concatenate images side by side
            combined_image = np.hstack((infra, color))

            # Add labels
            cv2.putText(combined_image, 'Infrared', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(combined_image, 'Color', (infra.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the combined image
            cv2.imshow('Infrared and Color Camera', combined_image)
        elif infra is not None:
            cv2.putText(infra, 'Infrared', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Infrared Camera', infra)
        elif color is not None:
            cv2.putText(color, 'Color', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Color Camera', color)

        # Handle key presses
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_event.set()
            break

        await asyncio.sleep(0.01)  # Yield control to the event loop


async def main():
    # Start Infrared capture coroutine
    infra_task = asyncio.create_task(capture_infrared())

    # Start Color capture in a separate thread
    color_thread = threading.Thread(target=capture_color, daemon=True)
    color_thread.start()

    # Start displaying frames
    display_task = asyncio.create_task(display_frames())

    # Wait for Infrared capture to finish
    await infra_task

    # Wait for display task to finish
    await display_task

    # Signal the Color capture thread to exit
    exit_event.set()
    color_thread.join()

    # Clean up OpenCV windows
    cv2.destroyAllWindows()

# Run the async main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit_event.set()
        print("Application interrupted by user.")
