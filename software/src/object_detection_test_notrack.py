import cv2
import torch
from ultralytics import YOLO

# video_path = "software/src/application/videos/basketball.mp4"
video_path = "software/src/application/videos/football.mp4"
# video_path = "software/src/application/videos/traffic.mp4"
video_path = '/Users/fredmac/Documents/DTU-FredMac/archive/Berghouse Leopard Jog.mp4'
video_path = '/Users/fredmac/Documents/DTU-FredMac/archive/DJI_0574.MP4' # bicycle riding

# HYPERPARAMETERS
model_name = "yolov8s.pt"
conf_threshold = 0.1       # Lower confidence threshold to increase recall
iou_threshold = 0.50       # Slightly higher IoU threshold to help with tighter boxes
MAX_FRAMES = 200
ALPHA = 0.7                 # (Unused without tracking)

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load YOLOv8 model without explicitly moving it to device
    model = YOLO(model_name)
    print(f"Model loaded: {model}")

    if model is None:
        print("Failed to load model. Check the model path or file integrity.")
        return

    # Use YOLOv8's prediction API, specifying device during prediction
    results = model.predict(source=video_path, device=device, conf=conf_threshold, iou=iou_threshold, stream=True)

    all_frames = []
    all_detections = []
    frame_count = 0

    for result in results:
        frame_count += 1
        if frame_count > MAX_FRAMES:
            break

        frame = result.orig_img  # original frame
        frame_detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in TARGET_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                default_label = model.names.get(class_id, f"class_{class_id}")
                display_label = CUSTOM_LABELS.get(class_id, default_label)

                frame_detections.append((x1, y1, x2, y2, display_label, conf))

        all_frames.append(frame.copy())
        all_detections.append(frame_detections)

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
