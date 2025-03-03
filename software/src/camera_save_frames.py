"""
Standalone Camera Recorder

This script captures frames from an RTSP camera stream and saves them to video files.
Each file is exactly 40 seconds long.

Configuration:
- Set CONTINUE_SAVING_TO_NEW_FILES to True to continuously record new 40-second files.
- Set it to False to record only one 40-second file.
- The script uses the same capture method (cv2.VideoCapture with an RTSP URL) as in the ROS2 node.
"""

import cv2
import time

# Configuration toggle: set to True for continuous recording, False for one file.
CONTINUE_SAVING_TO_NEW_FILES = True

# RTSP URL for the camera feed.
rtsp_url = "rtsp://192.168.145.25:8554/main.264"

# Duration (in seconds) for each video file.
VIDEO_DURATION_SECONDS = 40

def main():
    # Open the RTSP camera stream.
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}.")
        return

    # Attempt to get the camera's FPS; if unavailable, default to 10.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    print(f"Using FPS: {fps}")

    # Retrieve frame dimensions.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame size: {frame_width}x{frame_height}")

    file_counter = 1

    while True:
        # Create a new filename for each 40-second video.
        filename = f"output_{file_counter}.avi"
        # Define the codec and create VideoWriter object.
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        print(f"Recording to file: {filename}")

        start_time = time.time()
        while (time.time() - start_time) < VIDEO_DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.")
                break  # Optionally, you could try to reinitialize the stream here.
            out.write(frame)
            # Sleep to maintain roughly the desired FPS.
            time.sleep(1 / fps)
        out.release()
        print(f"Finished recording file: {filename}")

        # If we are not in continuous mode, break after one file.
        if not CONTINUE_SAVING_TO_NEW_FILES:
            break

        file_counter += 1

    cap.release()
    print("Recording complete. Camera stream released.")

if __name__ == '__main__':
    main()