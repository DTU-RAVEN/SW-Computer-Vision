import json
import cv2
import torch
from object_detection_utils import initialize_model, parse_detection_result
import rclpy                        
from rclpy.node import Node        
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge     

# Import Detection2DArray and related messages from vision_msgs and geometry_msgs
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, BoundingBox2D, Detection2DArray
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header

"""
Object Detection Node
====================
This module implements a ROS2 node for real-time object detection using YOLOv8.
It processes incoming camera frames, performs object detection, tracks objects
across frames, and publishes detection results as a Detection2DArray message.
"""

# ---------------------------
# HYPERPARAMETERS
# ---------------------------
model_name = "yolov8s.pt"
conf_threshold = 0.1
iou_threshold = 0.50
MAX_MISSES = 1
ALPHA = 1

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
    """
    ROS2 Node for Object Detection and Tracking
    
    This node subscribes to camera images, performs object detection,
    implements basic object tracking, and publishes detection results as
    a Detection2DArray message.
    """
    def __init__(self) -> None:
        super().__init__('object_detection_node')

        self.subscription = self.create_subscription(
            Image,            # Subscribe to sensor_msgs/Image messages
            '/camera/image',  # Topic name
            self.listener_callback,  # Callback function
            10               # QoS history depth
        )

        # Publisher now uses Detection2DArray message type
        self.publisher_ = self.create_publisher(Detection2DArray, '/vision/object_spotted', 10)
        self.bridge = CvBridge()

        # Initialize tracking state
        self.next_track_id = 0

        # Decide on compute device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        self.model = initialize_model(model_name, self.device)
        self.get_logger().info("Object Detection Node started. Subscribed to /camera/image.")

        # For object tracking
        self.track_history = {}

    def listener_callback(self, ros_image):
        """
        Process incoming camera frames for object detection and tracking,
        then publish all detections as a single Detection2DArray message.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV image: {e}")
            return

        stamp = ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9
        timestamp_ms = int(stamp * 1000)

        # Convert image from BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection using the YOLO model
        results = self.model.predict(
            rgb_frame,
            device=self.device,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Parse detections using the provided helper
        raw_detections = parse_detection_result(
            results[0], TARGET_CLASS_IDS, CUSTOM_LABELS, self.model
        )
        # raw_detections: list of (class_id, conf, x1, y1, x2, y2, display_label)

        # ---------------------
        # Tracking Logic
        # ---------------------
        current_detections = []
        for (class_id, conf, x1, y1, x2, y2, display_label) in raw_detections:
            track_id = self.assign_track_id(x1, y1, x2, y2)

            # Apply smoothing if the track already exists
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

        # Increase miss_count for tracks not detected in the current frame
        for t_id in list(self.track_history.keys()):
            if t_id not in current_detections:
                self.track_history[t_id]["miss_count"] += 1
                if self.track_history[t_id]["miss_count"] > MAX_MISSES:
                    del self.track_history[t_id]

        # ---------------------
        # Build and Publish Detection2DArray message
        # ---------------------
        detection_array_msg = Detection2DArray()
        # Use the original image header for timestamp and frame_id
        detection_array_msg.header = ros_image.header

        for t_id, info in self.track_history.items():
            if info["miss_count"] <= MAX_MISSES:
                x1, y1, x2, y2 = info["bbox"]
                conf = info["conf"]
                label = info["label"]

                # Compute bounding box center and size
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                detection_msg = Detection2D()
                detection_msg.header = ros_image.header

                # Populate the bounding box
                bbox = BoundingBox2D()
                bbox.center = Pose2D(x=cx, y=cy, theta=0.0)
                bbox.size_x = float(width)
                bbox.size_y = float(height)
                detection_msg.bbox = bbox

                # Populate the hypothesis with detection result
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = f"{t_id}:{label}"  # Combine track ID and label
                hypothesis.score = conf            # Confidence score
                detection_msg.results.append(hypothesis)

                detection_array_msg.detections.append(detection_msg)

        self.publisher_.publish(detection_array_msg)
        self.get_logger().info(f"Published Detection2DArray with {len(detection_array_msg.detections)} detections at t={timestamp_ms} ms.")

    def assign_track_id(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Associate the current detection with an existing track (using centroid distance)
        or create a new track if no matching track is found.
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
        Generate a unique tracking identifier.
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