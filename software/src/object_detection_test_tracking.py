"""
Real-time Object Detection and Tracking System

This module implements a real-time object detection and tracking system using YOLOv8 and ByteTrack.
It processes video input, detecting and tracking specific objects while implementing temporal
smoothing and object persistence across frames.

Key Features:
- Object detection using YOLOv8
- Object tracking using ByteTrack
- Temporal bounding box smoothing
- Object persistence across frame drops
- Interactive frame scrubbing interface
"""

import cv2
import torch
from ultralytics import YOLO

# video_path = "videos/basketball.mp4"
video_path = "videos/football.mp4"
# video_path = "videos/traffic.mp4"

"""
Configuration Parameters:
- model_name: YOLOv8 model variant to use
- conf_threshold: Minimum confidence score for detection acceptance
- iou_threshold: Intersection over Union threshold for detection merging
- MAX_FRAMES: Maximum number of frames to process
- MAX_MISSES: Number of frames to maintain tracking without detection
- ALPHA: Exponential smoothing factor for bounding box positions
"""

# HYPERPARAMETERS
model_name = "yolov8l.pt"
conf_threshold = 0.1       # Lower confidence threshold to increase recall
iou_threshold = 0.50        # Slightly higher IoU threshold to help with tighter boxes
MAX_FRAMES = 200
MAX_MISSES = 5              # How many frames to persist an object if it temporarily disappears
ALPHA = 0.7                 # Bounding box smoothing factor (0=no smoothing, 1=full smoothing)

"""
Target Classes:
Defines specific COCO dataset class IDs for detection filtering.
Each ID corresponds to objects of interest for this application.
"""

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

"""
Custom Labels:
Maps COCO class IDs to application-specific labels with additional
context about object specifications and requirements.
"""

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
    Visualize object detections on video frames.
    
    Args:
        frame (np.ndarray): Input video frame
        detections (list): List of detection tuples containing:
            (x1, y1, x2, y2, class_name, confidence, track_id)
    
    Returns:
        np.ndarray: Frame with annotated bounding boxes and labels
    """
    annotated_frame = frame.copy()
    for (x1, y1, x2, y2, class_name, conf, track_id) in detections:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {track_id}: {class_name} {conf:.2f}"
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
    Main execution function implementing the object detection and tracking pipeline.
    
    The pipeline consists of several key components:
    1. Device selection (MPS/CPU)
    2. Model initialization and configuration
    3. Frame-by-frame processing with tracking
    4. Temporal smoothing of bounding boxes
    5. Object persistence management
    6. Interactive visualization
    """
    
    # Device Selection
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Model Initialization
    model = YOLO(model_name).to(device)

    # Configure and initialize tracking
    tracking_results = model.track(
        source=video_path,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        tracker="config/bytetrack.yaml",  # ByteTrack configuration
        stream=True,
        show=False
    )

    """
    Tracking State Management:
    track_history maintains the state of tracked objects across frames:
    - bbox: Last known bounding box coordinates
    - conf: Detection confidence
    - label: Object class label
    - miss_count: Frames since last detection
    """
    track_history = {}
    
    all_frames = []
    all_detections = []
    frame_count = 0

    # track_history keeps track of each object's last-known bbox and how many times it has been missed
    # Format: track_id -> {"bbox": (x1, y1, x2, y2), "conf": float, "label": str, "miss_count": int}
    track_history = {}

    for result in tracking_results:
        """
        Frame Processing Loop:
        1. Extract and filter detections
        2. Apply temporal smoothing
        3. Update tracking history
        4. Handle missing detections
        5. Prepare visualization data
        """
        
        frame_count += 1
        if frame_count > MAX_FRAMES:
            break

        frame = result.orig_img  # original frame
        frame_h, frame_w = frame.shape[:2]

        # Current frame's detections, after filtering
        current_detections = []

        # If we get no detections, we just increment miss_count for all in track_history
        if result.boxes is not None and len(result.boxes) > 0:
            """
            Detection Processing:
            - Filter by target classes
            - Apply temporal smoothing
            - Update tracking history
            """
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in TARGET_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1

                default_label = model.names.get(class_id, f"class_{class_id}")
                display_label = CUSTOM_LABELS.get(class_id, default_label)

                # Check if we've seen this track_id before
                if track_id in track_history:
                    # Get the old bounding box for smoothing
                    old_x1, old_y1, old_x2, old_y2 = track_history[track_id]["bbox"]
                    # Simple bounding box smoothing
                    x1 = int(ALPHA * x1 + (1 - ALPHA) * old_x1)
                    y1 = int(ALPHA * y1 + (1 - ALPHA) * old_y1)
                    x2 = int(ALPHA * x2 + (1 - ALPHA) * old_x2)
                    y2 = int(ALPHA * y2 + (1 - ALPHA) * old_y2)

                # Update track_history for this track_id
                track_history[track_id] = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "label": display_label,
                    "miss_count": 0  # reset because it's detected this frame
                }

                current_detections.append(track_id)

        """
        Missing Object Management:
        - Increment miss counter for undetected objects
        - Remove objects exceeding MAX_MISSES threshold
        """
        # Increase miss_count for those not detected in this frame
        for t_id in list(track_history.keys()):
            if t_id not in current_detections:
                track_history[t_id]["miss_count"] += 1
                # If an object is missing, keep it up to MAX_MISSES frames
                if track_history[t_id]["miss_count"] > MAX_MISSES:
                    del track_history[t_id]  # remove from history
       
        """
        Visualization Data Preparation:
        Compile final list of objects to display, including:
        - Currently detected objects
        - Recently missing objects within MAX_MISSES threshold
        """
        # Build a list of final bounding boxes to draw (both newly seen and persisted)
        frame_detections = []
        for t_id, info in track_history.items():
            x1, y1, x2, y2 = info["bbox"]
            conf = info["conf"]
            label = info["label"]

            # We only want to draw if the object was either detected this frame
            # or has not exceeded MAX_MISSES.
            if info["miss_count"] <= MAX_MISSES:
                frame_detections.append((x1, y1, x2, y2, label, conf, t_id))

        all_frames.append(frame.copy())
        all_detections.append(frame_detections)

    """
    Interactive Visualization Interface:
    Creates a window with a trackbar for frame scrubbing
    """
    # Build a player interface to scrub frames
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    def on_trackbar(pos):
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
