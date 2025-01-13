import json
import cv2
import torch
# ---------------- NEW IMPORT ----------------
from object_detection_shared import initialize_model, parse_detection_result
# --------------------------------------------
# from ultralytics import YOLO  # Now handled by the helper script

import rclpy                        
from rclpy.node import Node        
from sensor_msgs.msg import Image  
from std_msgs.msg import String    
from cv_bridge import CvBridge     

# ---------------------------
# HYPERPARAMETERS
# ---------------------------
model_name = "yolov8l.pt"
conf_threshold = 0.01
iou_threshold = 0.50
MAX_FRAMES = 200   
MAX_MISSES = 5
ALPHA = 0.7

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

class ObjectDetectionNode(Node):
    """ @brief Object detection node class
    """
    def __init__(self) -> None:
        """ @brief Object detection node constructor
        """
        super().__init__('object_detection_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/vision/object_spotted', 10)
        self.bridge = CvBridge()

        ## Keep next track id
        self.next_track_id = 0

        # Decide on device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        # -------------- IMPORTANT CHANGE: Use our helper to load the YOLO model --------------
        self.model = initialize_model(model_name, self.device)

        self.get_logger().info("Object Detection Node started. Subscribed to /camera/image.")

        # For tracking
        self.track_history = {}

    def listener_callback(self, ros_image):
        """ @brief Callback for each incoming image message
        """
        # Convert ROS Image to OpenCV (BGR)
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV image: {e}")
            return

        stamp = ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9
        timestamp_ms = int(stamp * 1000)

        # Convert BGR -> RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.model.predict(
            rgb_frame,
            device=self.device,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # ---------------------------------------------
        # IMPORTANT CHANGE: Use parse_detection_result
        # ---------------------------------------------
        # results[0] is the YOLO result for this single frame
        raw_detections = parse_detection_result(
            results[0], TARGET_CLASS_IDS, CUSTOM_LABELS, self.model
        )
        # raw_detections is a list of (class_id, conf, x1, y1, x2, y2, display_label).

        # Tracking logic
        current_detections = []

        for (class_id, conf, x1, y1, x2, y2, display_label) in raw_detections:
            # Assign track ID (simple or more advanced)
            track_id = self.assign_track_id(x1, y1, x2, y2)

            # Smoothing if existing track
            if track_id in self.track_history:
                old_x1, old_y1, old_x2, old_y2 = self.track_history[track_id]["bbox"]
                x1 = int(ALPHA * x1 + (1 - ALPHA) * old_x1)
                y1 = int(ALPHA * y1 + (1 - ALPHA) * old_y1)
                x2 = int(ALPHA * x2 + (1 - ALPHA) * old_x2)
                y2 = int(ALPHA * y2 + (1 - ALPHA) * old_y2)

            self.track_history[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "label": display_label,
                "miss_count": 0
            }

            current_detections.append(track_id)

        # Increase miss_count for undetected in this frame
        for t_id in list(self.track_history.keys()):
            if t_id not in current_detections:
                self.track_history[t_id]["miss_count"] += 1
                if self.track_history[t_id]["miss_count"] > MAX_MISSES:
                    del self.track_history[t_id]

        # Prepare JSON results
        results_to_publish = []
        for t_id, info in self.track_history.items():
            if info["miss_count"] <= MAX_MISSES:
                x1, y1, x2, y2 = info["bbox"]
                conf = info["conf"]
                label = info["label"]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                detection_dict = {
                    "object_id": t_id,
                    "position": [cx, cy, width, height],
                    "label": label,
                    "confidence": conf,
                    "timestamp": timestamp_ms,
                }
                results_to_publish.append(detection_dict)

        msg = String()
        msg.data = json.dumps({"detections": results_to_publish})
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published {len(results_to_publish)} detections at t={timestamp_ms} ms.")

    def assign_track_id(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Assign a unique track ID based on bounding box position.
        This is a placeholder for a more robust tracker like SORT/DeepSORT.
        """
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for t_id, info in self.track_history.items():
            old_x1, old_y1, old_x2, old_y2 = info["bbox"]
            old_cx = (old_x1 + old_x2) // 2
            old_cy = (old_y1 + old_y2) // 2

            distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            if distance < max(x2 - x1, y2 - y1) * 0.5:
                return t_id

        return self.get_new_track_id()

    def get_new_track_id(self) -> int:
        """
        Generate a new unique track ID.
        """
        if not hasattr(self, 'next_track_id'):
            self.next_track_id = 0
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

def main(args=None) -> None:
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
