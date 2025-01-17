import cv2
import torch
# ---------------- NEW IMPORT ----------------
from object_detection_shared import initialize_model, parse_detection_result
# --------------------------------------------

# ---------------- TOGGLE --------------------
USE_WEBCAM = False  # Set to True to use webcam instead of a video file.
# -------------------------------------------------

# Paths to potential video files; only used if USE_WEBCAM=False
video_path = '/Users/fredmac/Documents/DTU-FredMac/Drone/archive/Berghouse.mp4'

# HYPERPARAMETERS
model_name = "yolov8s.pt"
conf_threshold = 0.1       # Lower confidence threshold to increase recall
iou_threshold = 0.50       # Slightly higher IoU threshold to help with tighter boxes
MAX_FRAMES = 200
ALPHA = 0.7                # (Unused without tracking)

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
    Draw bounding boxes and labels on a copy of the frame based on detections.
    detections: list of tuples (x1, y1, x2, y2, class_name, conf)
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
    print("Starting object detection test...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = initialize_model(model_name, device)
    print(f"Model loaded: {model}")

    if model is None:
        print("Failed to load model. Check the model path or file integrity.")
        return

    # Create a window for display
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    if USE_WEBCAM:
        # ------------------ WEBCAM FEED ------------------
        print("Using webcam feed...")
        cap = cv2.VideoCapture(0)  # 0 means default webcam; change if needed

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam. Exiting.")
                break

            # Run detection on the current frame
            results = model.predict(
                source=frame,
                device=device,
                conf=conf_threshold,
                iou=iou_threshold
            )

            # The result for a single image/frame is just results[0]
            raw_detections = parse_detection_result(
                results[0], TARGET_CLASS_IDS, CUSTOM_LABELS, model
            )

            # Convert to the format (x1, y1, x2, y2, class_name, conf)
            frame_detections = [
                (x1, y1, x2, y2, label, conf)
                for (class_id, conf, x1, y1, x2, y2, label) in raw_detections
            ]

            # Draw detections and show
            annotated_frame = draw_detections(frame, frame_detections)
            cv2.imshow("Detections", annotated_frame)

            # Press ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # ------------------ VIDEO FILE -------------------
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
