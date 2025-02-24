#!/usr/bin/env python3
import cv2
import threading
from flask import Flask, Response

# Initialize the Flask app
app = Flask(__name__)

# Global variable to hold the latest detection frame and a lock for thread safety.
outputFrame = None
lock = threading.Lock()

def detection_loop():
    """
    This function simulates the detection pipeline.
    Replace the dummy detection code below with your actual detection code.
    It reads from the camera, applies detections (e.g. drawing bounding boxes),
    and then updates the global outputFrame variable.
    """
    global outputFrame
    # For example, open the camera (adjust the index or use your detection video source)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ---------------------------
        # Replace the block below with your object detection logic.
        # For demonstration, we simply draw a green rectangle.
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        # ---------------------------

        # Update the global frame in a thread-safe way.
        with lock:
            outputFrame = frame.copy()

def generate_stream():
    """
    Generator function that encodes the latest frame as JPEG and yields it in a multipart response.
    This is a common method to stream video over HTTP (MJPEG).
    """
    global outputFrame
    while True:
        with lock:
            if outputFrame is None:
                continue
            # Encode the frame in JPEG format.
            flag, encodedImage = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        # Yield the output frame in byte format with the correct multipart header.
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' +
              encodedImage.tobytes() +
              b'\r\n')

@app.route("/video_feed")
def video_feed():
    """
    Flask route to serve the video stream. Simply point your browser or MJPEG client to:
      http://<jetson_ip>:5000/video_feed
    """
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # Start the detection loop in a separate daemon thread.
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    # Start the Flask app on all interfaces at port 5000.
    app.run(host="0.0.0.0", port=5000, threaded=True)