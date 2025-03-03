#!/usr/bin/env python3
"""
Segmented RTSP Recorder with Full Restart

Each time a new video segment is recorded (set to 40 seconds),
the capture process is completely shut down and restarted.
Before opening the RTSP stream, the network interface is reconfigured.
"""

import cv2
import time
import subprocess

# Configuration
CONTINUE_SAVING_TO_NEW_FILES = True  # Set False to record only one segment
VIDEO_DURATION_SECONDS = 40          # Duration of each segment in seconds
rtsp_url = "rtsp://192.168.145.25:8554/main.264"

# Network interface settings (change as needed)
NET_INTERFACE = "eth0"
IP_ADDRESS = "192.168.145.10/24"

def reconfigure_network():
    """Run network reconfiguration commands to ensure the interface is up."""
    commands = [
        ["sudo", "ip", "addr", "flush", "dev", NET_INTERFACE],
        ["sudo", "ip", "addr", "add", IP_ADDRESS, "dev", NET_INTERFACE],
        ["sudo", "ip", "link", "set", NET_INTERFACE, "up"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Network command {' '.join(cmd)} failed: {result.stderr.decode().strip()}")
        else:
            print(f"Executed: {' '.join(cmd)}")
    # Give a short delay for the interface to settle.
    time.sleep(1)

def record_segment(segment_number):
    """Open the RTSP stream, record for a fixed duration, and then close the stream."""
    print(f"\n--- Starting segment {segment_number} ---")
    # Reconfigure network before starting capture
    reconfigure_network()

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return False

    # Get capture properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Segment {segment_number}: Using FPS: {fps}, Frame size: {frame_width}x{frame_height}")

    # Initialize VideoWriter for this segment
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    filename = f"output_{segment_number}.avi"
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: VideoWriter did not open.")
        cap.release()
        return False
    print(f"Recording to file: {filename}")

    start_time = time.time()
    frames_recorded = 0
    while time.time() - start_time < VIDEO_DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame; continuing...")
            # We may want to sleep a bit before retrying
            time.sleep(0.05)
            continue
        out.write(frame)
        frames_recorded += 1

    print(f"Segment {segment_number} finished. Recorded {frames_recorded} frames.")
    cap.release()
    out.release()
    return True

def main():
    segment_number = 1
    while True:
        if not record_segment(segment_number):
            print("Segment recording failed. Retrying in 2 seconds...")
            time.sleep(2)
            continue

        # If only one segment is required, break after the first one.
        if not CONTINUE_SAVING_TO_NEW_FILES:
            break
        segment_number += 1

if __name__ == "__main__":
    main()