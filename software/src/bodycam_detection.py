import cv2
import torch
from ultralytics import YOLO

# Feel free to change this to any integer value you like for quick testing.
MAX_FRAMES = 200

# Define the set of COCO class IDs that approximate your desired categories
TARGET_CLASS_IDS = {
    0,   # person (for person/mannequin)
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

# Optionally, define a custom label dictionary if you want user-friendly names.
# (However, for drawing on frames, you can just use YOLO's built-in label strings.)
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
    Draw bounding boxes and labels on a copy of the frame based on filtered detections.
    detections: list of tuples (x1, y1, x2, y2, class_name, conf)
    """
    annotated_frame = frame.copy()
    for x1, y1, x2, y2, class_name, conf in detections:
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
    # Select 'mps' device if available, otherwise fall back to CPU.
    # (This uses Apple's Metal Performance Shaders on macOS with Apple Silicon.)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load YOLOv8 model (pretrained on COCO)
    model = YOLO("yolov8n.pt").to(device)

    # Open the video file
    cap = cv2.VideoCapture("run.MP4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    all_frames = []
    all_detections = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or can't read further

        frame_count += 1
        if frame_count > MAX_FRAMES:
            break

        # Convert BGR -> RGB if needed. YOLOv8 can handle BGR directly,
        # but converting is often the canonical approach.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection on the current frame
        results = model.predict(rgb_frame, device=device, verbose=False)
        boxes = results[0].boxes

        # Filter boxes to keep only TARGET_CLASS_IDS
        frame_detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id in TARGET_CLASS_IDS:
                conf = float(box.conf[0])
                # Convert xyxy floats to integers for drawing
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Use YOLO's default label or a custom label
                default_label = model.names[class_id]
                display_label = CUSTOM_LABELS.get(class_id, default_label)

                frame_detections.append((x1, y1, x2, y2, display_label, conf))

        # Store original frame (for later display) and the detections
        all_frames.append(frame.copy())
        all_detections.append(frame_detections)

    cap.release()

    # Now that we've processed the video (up to MAX_FRAMES), build a player interface
    # with a trackbar to scrub frames/detections.
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    
    # Add state variables for playback control
    playing = False
    last_frame_time = 0
    fps = 30  # Default playback speed
    
    def on_trackbar(pos):
        # Get the corresponding frame and its detections
        frame_idx = pos
        if 0 <= frame_idx < len(all_frames):
            frame = all_frames[frame_idx]
            detections = all_detections[frame_idx]
            annotated = draw_detections(frame, detections)
            
            # Add play/pause indicator
            status = "Playing" if playing else "Paused"
            cv2.putText(
                annotated,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Detections", annotated)

    total_frames = len(all_frames)
    if total_frames == 0:
        print("No frames processed. Exiting.")
        cv2.destroyAllWindows()
        return

    # Create trackbar
    cv2.createTrackbar("Frame", "Detections", 0, total_frames - 1, on_trackbar)
    
    def get_current_frame():
        return cv2.getTrackbarPos("Frame", "Detections")
    
    def set_frame(frame_idx):
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        cv2.setTrackbarPos("Frame", "Detections", frame_idx)

    # Initial display
    on_trackbar(0)

    while True:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Handle automatic playback
        if playing and (current_time - last_frame_time) > 1.0/fps:
            next_frame = get_current_frame() + 1
            if next_frame >= total_frames:
                playing = False  # Stop at end of video
            else:
                set_frame(next_frame)
                last_frame_time = current_time

        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):  # Esc key or 'q'
            break
        elif key == ord(' '):  # Space bar - toggle play/pause
            playing = not playing
            last_frame_time = current_time
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            playing = False
            set_frame(get_current_frame() + 1)
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            playing = False
            set_frame(get_current_frame() - 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
