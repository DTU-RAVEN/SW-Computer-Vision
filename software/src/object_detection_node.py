import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from sensor_msgs.msg import Image   # type: ignore
from std_msgs.msg import String     # type: ignore
from cv_bridge import CvBridge      # type: ignore

import cv2
import torch
from ultralytics import YOLO
import json

# ---------------------------
# HYPERPARAMETERS
# ---------------------------
model_name = "yolov8l.pt"
conf_threshold = 0.01       # Lower confidence threshold to increase recall
iou_threshold = 0.50        # Slightly higher IoU threshold to help with tighter boxes
MAX_FRAMES = 200            # Not used in ROS node but kept for consistency
MAX_MISSES = 5              # How many frames to persist an object if it temporarily disappears
ALPHA = 0.7                 # Bounding box smoothing factor (0=no smoothing, 1=full smoothing)

# Define the set of COCO class IDs that approximate the desired categories
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
    59,  # bed (for mattress)
}

# Optionally, define a custom label dictionary if you want user-friendly names.
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

class ObjectDetectionNode(Node):
    """ @brief Object detection node class
    """
    def __init__(self) -> None:
        """ @brief Object detection node constructor
        """
        super().__init__('object_detection_node')

        ## Detection node subscription instance. Subscribing to /camera/image
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',

            ## Listener callback
            self.listener_callback,
            10
        )

        # Publisher for the detection results (JSON-encoded string on /vision/object_spotted)
        self.publisher_ = self.create_publisher(String, '/vision/object_spotted', 10)

        ## CvBridge for image conversion
        self.bridge = CvBridge()

        ## Device to do matrix multiplications
        self.device = None
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        ## YOLOv8 model (pretrained on COCO) to the chosen device
        self.model = YOLO(model_name).to(self.device)

        self.get_logger().info("Object Detection Node started. Subscribed to /camera/image.")

        # ---------------------------
        # Tracking Setup
        # ---------------------------
        # track_history keeps track of each object's last-known bbox and how many times it has been missed
        # Format: track_id -> {"bbox": (x1, y1, x2, y2), "conf": float, "label": str, "miss_count": int}
        self.track_history : dict[int, dict]

    def listener_callback(self, ros_image):
        """ @brief Callback that runs each time we receive an Image on /camera/image.
            @ros_image The image
        """

        # Convert ROS Image to OpenCV format (BGR)
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV image: {e}")
            return

        # Retrieve the timestamp (from the image header).
        # Convert ROS2 time to float in seconds or to an int in milliseconds.
        stamp = ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9
        timestamp_ms = int(stamp * 1000)  # example in milliseconds

        # --------------------------------------------------------
        # Perform object detection using your YOLO approach
        # --------------------------------------------------------
        # Convert BGR -> RGB as YOLOv8 expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection with confidence and IOU thresholds
        results = self.model.predict(
            rgb_frame, 
            device=self.device, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False
        )
        boxes = results[0].boxes

        # Current frame's detections after filtering and tracking
        current_detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in TARGET_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Here, since we're not using an external tracker, assign a temporary track_id
                # You might want to implement a better tracking mechanism (e.g., SORT, Deep SORT)
                track_id = self.assign_track_id(x1, y1, x2, y2)

                default_label = self.model.names.get(class_id, f"class_{class_id}")
                display_label = CUSTOM_LABELS.get(class_id, default_label)

                # Check if we've seen this track_id before for smoothing
                if track_id in self.track_history:
                    old_x1, old_y1, old_x2, old_y2 = self.track_history[track_id]["bbox"]
                    # Simple bounding box smoothing
                    x1 = int(ALPHA * x1 + (1 - ALPHA) * old_x1)
                    y1 = int(ALPHA * y1 + (1 - ALPHA) * old_y1)
                    x2 = int(ALPHA * x2 + (1 - ALPHA) * old_x2)
                    y2 = int(ALPHA * y2 + (1 - ALPHA) * old_y2)

                # Update track_history for this track_id
                self.track_history[track_id] = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "label": display_label,
                    "miss_count": 0  # reset because it's detected this frame
                }

                current_detections.append(track_id)

        # Increase miss_count for those not detected in this frame
        for t_id in list(self.track_history.keys()):
            if t_id not in current_detections:
                self.track_history[t_id]["miss_count"] += 1
                # If an object is missing, keep it up to MAX_MISSES frames
                if self.track_history[t_id]["miss_count"] > MAX_MISSES:
                    del self.track_history[t_id]  # remove from history

        # Build a list of final bounding boxes to publish (both newly seen and persisted)
        results_to_publish = []
        for t_id, info in self.track_history.items():
            x1, y1, x2, y2 = info["bbox"]
            conf = info["conf"]
            label = info["label"]

            # Only include if not exceeded miss count
            if info["miss_count"] <= MAX_MISSES:
                # Calculate center, width, and height
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                # Construct a detection dictionary consistent with your specs
                detection_dict = {
                    "object_id": t_id,               # Unique track ID
                    "position": [cx, cy, width, height],
                    "label": label,
                    "confidence": conf,
                    "timestamp": timestamp_ms,
                }
                results_to_publish.append(detection_dict)

        # Publish the entire list of detections as a JSON string
        msg = String()
        msg.data = json.dumps({"detections": results_to_publish})
        self.publisher_.publish(msg)

        # Optional debug log
        self.get_logger().info(f"Published {len(results_to_publish)} detections at t={timestamp_ms} ms.")

    def assign_track_id(self, x1 : int, y1 : int, x2 : int, y2 : int) -> int:
        """
        Assign a unique track ID based on the bounding box position.
        This is a placeholder for a more robust tracking algorithm.
        """
        # Simple centroid-based assignment
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for t_id, info in self.track_history.items():
            old_x1, old_y1, old_x2, old_y2 = info["bbox"]
            old_cx = (old_x1 + old_x2) // 2
            old_cy = (old_y1 + old_y2) // 2

            distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            if distance < max(x2 - x1, y2 - y1) * 0.5:
                return t_id

        # If no existing track is close, assign a new ID
        return self.get_new_track_id()

    def get_new_track_id(self) -> int:
        """
        Generate a new unique track ID.
        """
        if not hasattr(self, 'next_track_id'):
            ##  Find next track id
            self.next_track_id = 0
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

def main(args: list[str] = []) -> None:
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Object Detection node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
