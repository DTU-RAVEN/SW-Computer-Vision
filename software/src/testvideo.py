#!/usr/bin/env python3
import cv2

video_path = "software/src/videos/football.mp4"

# Attempt to open the video file.
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video file:", video_path)
else:
    print("Video file opened successfully.")
    # Try to read one frame.
    ret, frame = cap.read()
    if ret:
        print("Frame read successfully.")
    else:
        print("Failed to read a frame.")

cap.release()