from winsdk.windows.media.capture import MediaCapture, MediaCaptureInitializationSettings, MediaCaptureSharingMode, MediaCaptureMemoryPreference, StreamingCaptureMode
from winsdk.windows.media.devices import MediaDeviceControl

import asyncio


async def control_ir_led():
    media_capture = MediaCapture()
    await media_capture.initialize_async()

    # Attempt to access the torch control (if available)
    video_device_controller = media_capture.video_device_controller

    try:
        print(dir(video_device_controller))
        torch_control = video_device_controller.torch_control
        print(torch_control.supported)
        if torch_control.supported:
            print("Torch is supported. Turning it on.")
            print(torch_control.power)
        else:
            print("Torch control is not supported on this device.")
    except AttributeError:
        print("Torch control is not available on this device.")

    # Cleanup
    media_capture.close()

asyncio.run(control_ir_led())
