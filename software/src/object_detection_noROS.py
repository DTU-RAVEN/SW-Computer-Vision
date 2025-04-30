import cv2
import torch
import gi
import numpy as np
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst
from utils.object_detection_utils import initialize_model, parse_detection_result

"""
Object Detection Test Module

This module implements real-time object detection using YOLOv8 model with support for both
webcam and video file inputs. It includes custom class filtering and visualization features.

Key Features:
- Supports both webcam and video file input sources
- Hardware acceleration (CUDA/MPS) when available
- Custom class filtering with specific target categories
- Real-time visualization with confidence scores
- Video scrubbing interface for file playback
"""

# Configuration flags
USE_VIDEO_FILE = False  # True → play from video file; False → read live camera RTP stream

# File paths and model configuration
video_path = '/Users/fredmac/Documents/DTU-FredMac/Drone/archive/Berghouse.mp4'

# Model and detection parameters
model_name = "yolov8s.pt"
conf_threshold = 0.1       # Confidence threshold for detection filtering
iou_threshold = 0.50      # Intersection over Union threshold for NMS
MAX_FRAMES = 200          # Maximum number of frames to process from video
ALPHA = 0.7              # Transparency factor (reserved for tracking features)

# Define the set of COCO class IDs that approximate your desired categories
TARGET_CLASS_IDS = {
    0,   # person
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    8,   # boat
    11,  # stop sign
    25,  # umbrella
    28,  # suitcase
    30,  # skis
    31,  # snowboard
    32,  # sports ball
    34,  # baseball bat
    38,  # tennis racket
    59,  # bed
}

CUSTOM_LABELS = {
    0:  "Person / Mannequin",
    2:  "Car (>1:8 Scale Model)",
    3:  "Motorcycle (>1:8 Scale Model)",
    4:  "Airplane (>3m Wing Span Scale Model)",
    5:  "Bus (>1:8 Scale Model)",
    8:  "Boat (>1:8 Scale Model)",
    11: "Stop Sign (Flat, Upwards Facing)",
    25: "Umbrella",
    28: "Suitcase",
    30: "Skis",
    31: "Snowboard",
    32: "Sports Ball (Regulation Size)",
    34: "Baseball Bat",
    38: "Tennis Racket",
    59: "Bed / Mattress (> Twin Size)",
}


def draw_detections(frame, detections):
    """
    Visualizes object detections on a frame with bounding boxes and labels.

    Args:
        frame (np.ndarray): Input frame/image
        detections (list): List of tuples containing detection data
                          (x1, y1, x2, y2, class_name, confidence)

    Returns:
        np.ndarray: Frame with drawn detections
    """
    annotated_frame = frame.copy()
    for (x1, y1, x2, y2, class_name, conf) in detections:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return annotated_frame


def main():
    """
    Main execution function for object detection.
    
    Workflow:
    1. Sets up hardware device (CUDA/MPS/CPU)
    2. Initializes YOLOv8 model
    3. Processes input source (webcam/video)
    4. Performs real-time detection and visualization
    5. Handles user interaction and display
    """
    print("Starting object detection test...")

    # Device selection logic - prioritize GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device}")

    # Initialize the model
    model = initialize_model(model_name, device)
    print(f"Model loaded")

    if model is None:
        print("Failed to load model. Check the model path or file integrity.")
        return

    # Create a window for display
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    if not USE_VIDEO_FILE:
        """
        Camera Stream Processing (RTP H.264 over UDP)

        Receives a live stream on UDP port 5000, decodes it with GStreamer,
        and performs real‑time detection and visualization.
        """
        print("Using live camera stream via GStreamer…")

        # Build and start the GStreamer pipeline
        Gst.init(None)
        pipeline = Gst.parse_launch(
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'clock-rate=90000,encoding-name=H264,payload=96" ! '
            'rtpjitterbuffer ! rtph264depay ! decodebin ! videoconvert ! '
            'video/x-raw,format=BGR ! appsink name=sink '
            'emit-signals=true sync=false max-buffers=1 drop=true'
        )
        sink = pipeline.get_by_name("sink")
        pipeline.set_state(Gst.State.PLAYING)

        try:
            while True:
                sample = sink.emit("try-pull-sample", 100_000_000)  # 100 ms
                if sample is None:
                    continue

                buf = sample.get_buffer()
                caps = sample.get_caps()
                width = caps.get_structure(0).get_value("width")
                height = caps.get_structure(0).get_value("height")

                ok, mapinfo = buf.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                frame = np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3))
                buf.unmap(mapinfo)

                # Run YOLO v8 detection
                results = model.predict(
                    source=frame,
                    device=device,
                    conf=conf_threshold,
                    iou=iou_threshold
                )
                raw_detections = parse_detection_result(
                    results[0], TARGET_CLASS_IDS, CUSTOM_LABELS, model
                )
                frame_detections = [
                    (x1, y1, x2, y2, label, conf)
                    for (class_id, conf, x1, y1, x2, y2, label) in raw_detections
                ]

                annotated_frame = draw_detections(frame, frame_detections)
                cv2.imshow("Detections", annotated_frame)

                # ESC to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()

    else:
        """
        Video File Processing
        
        Processes a video file in batches, storing frames and detections
        for interactive playback. Includes a scrubbing interface for
        frame navigation.
        """
        print(f"Using video file: {video_path}")
        # Use YOLOv8's streaming API on a video source
        results = model.predict(
            source=video_path,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            stream=True
        )

        all_frames = []
        all_detections = []
        frame_count = 0

        for result in results:
            print(f"Processing frame {frame_count}...")
            frame_count += 1
            if frame_count > MAX_FRAMES:
                break

            frame = result.orig_img  # original frame

            # Parse detection result
            raw_detections = parse_detection_result(
                result, TARGET_CLASS_IDS, CUSTOM_LABELS, model
            )
            # Convert to the format (x1, y1, x2, y2, class_name, conf)
            frame_detections = [
                (x1, y1, x2, y2, label, conf)
                for (class_id, conf, x1, y1, x2, y2, label) in raw_detections
            ]

            all_frames.append(frame.copy())
            all_detections.append(frame_detections)

        # Build a player interface to scrub frames
        def on_trackbar(pos):
            """
            Callback function for the video scrubbing trackbar.
            
            Args:
                pos (int): Current position in the video timeline
            """
            if 0 <= pos < len(all_frames):
                frame = all_frames[pos]
                detections = all_detections[pos]
                annotated = draw_detections(frame, detections)
                cv2.imshow("Detections", annotated)

        total_frames = len(all_frames)
        if total_frames == 0:
            print("No frames processed. Exiting.")
            cv2.destroyAllWindows()
            return

        cv2.createTrackbar("Frame", "Detections", 0, total_frames - 1, on_trackbar)
        on_trackbar(0)

        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
