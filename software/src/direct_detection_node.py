#!/usr/bin/env python3
"""
Camera and Detection Node

This node:
- Opens a video source (either an RTSP stream or a local MP4 file).
- Reads frames directly (at 10Hz) without publishing them on a ROS topic.
- Performs object detection and simple tracking on each frame using YOLOv8.
- Publishes the detection results as a Detection2DArray message on the topic '/vision/object_spotted'.

Configuration:
    - USE_MP4_FILE: Toggle between using a video file or an RTSP stream.
    - video_path: Path to the MP4 file (if USE_MP4_FILE is True).
    - rtsp_url: URL of the RTSP stream (if USE_MP4_FILE is False).
"""
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, BoundingBox2D, Detection2DArray
import torch

# Import helper functions for model initialization and result parsing.
from utils.object_detection_utils import initialize_model, parse_detection_result

# ---------------------------
# GLOBAL CONFIGURATION
# ---------------------------
USE_MP4_FILE = False  # Set to True to use video file input; False to use RTSP stream.
video_path = "software/src/videos/categories/tennis racket.mov"
rtsp_url = "rtsp://192.168.153.1:8899/stream1"

PUBLISH_FRAMES_WITH_DETECTIONS = True  # Set to True to publish frames with detection overlays on '/vision/detection_frames'

# Additional imports for publishing detection frames.
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# ---------------------------
# DETECTION AND TRACKING PARAMETERS
# ---------------------------
model_name = "yolov8s.pt"
conf_threshold = 0.1
iou_threshold = 0.50
MAX_MISSES = 1
ALPHA = 1  # Smoothing factor (set to 1 to use the current detection only)

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
    2:  "Car",
    3:  "Motorcycle",
    4:  "Airplane",
    5:  "Bus",
    8:  "Boat",
    11: "Stop Sign",
    25: "Umbrella",
    28: "Suitcase",
    30: "Skis",
    31: "Snowboard",
    32: "Sports Ball",
    34: "Baseball Bat",
    38: "Tennis Racket",
    59: "Bed / Mattress",
}

class CombinedDetectionNode(Node):
    """
    A combined ROS2 node that performs both camera frame acquisition and object detection.
    """
    def __init__(self, model):
        super().__init__('combined_detection_node')
        self.model = model

        # Publisher for Detection2DArray messages.
        self.publisher_ = self.create_publisher(Detection2DArray, '/vision/object_spotted', 10)

        # Optionally create publisher for detection frames.
        if PUBLISH_FRAMES_WITH_DETECTIONS:
            self.detection_frame_pub = self.create_publisher(Image, '/vision/detection_frames', 10)
            self.bridge = CvBridge()
            self.get_logger().info("Detection frame publisher initialized.")

        # Timer for periodic frame capture and processing (10Hz).
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize the video capture source.
        if USE_MP4_FILE:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.get_logger().error("Could not open video file.")
            else:
                self.get_logger().info("Opened video file for input.")
        else:
            # Build GStreamer pipeline for RTSP
            pipeline_str = (
                f"rtspsrc location={rtsp_url} "
                "protocols=tcp latency=200 ! "
                "rtpjitterbuffer latency=200 ! "
                "rtph264depay ! "
                "h264parse ! "
                "avdec_h264 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "queue leaky=downstream max-size-buffers=1 ! "
                "appsink name=sink "
                "emit-signals=true sync=false drop=true"
            )
            try:
                self.pipeline = Gst.parse_launch(pipeline_str)
                self.sink = self.pipeline.get_by_name("sink")
                self.pipeline.set_state(Gst.State.PLAYING)
                self.get_logger().info(f"Opened RTSP stream from {rtsp_url} with GStreamer.")
            except Exception as e:
                self.get_logger().error(f"Failed to open RTSP stream: {e}")
                self.pipeline = None
                self.sink = None

        # Initialize object detection model.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        self.model = initialize_model(model_name, self.device)
        self.get_logger().info("Object detection model initialized.")

        # Tracking state.
        self.track_history = {}
        self.next_track_id = 0

    def timer_callback(self):
        """
        Timer callback that:
        1. Captures a frame from the camera/video.
        2. Performs object detection and tracking.
        3. Publishes the detection results.
        """
        if USE_MP4_FILE:
            ret, frame = self.cap.read()
            # If using video file, loop the video upon reaching its end.
            if not ret:
                self.get_logger().warning("Reached end of video. Looping back to start.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning("Failed to read frame from video file.")
                return
        else:
            if self.sink is None:
                self.get_logger().warning("GStreamer sink not initialized.")
                return
            sample = self.sink.emit("try-pull-sample", 100_000_000)  # 100 ms
            if sample is None:
                self.get_logger().warning("No sample received from RTSP stream.")
                return
            buf = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")
            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                self.get_logger().warning("Failed to map GStreamer buffer.")
                return
            frame = np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3))
            buf.unmap(mapinfo)

        # Create a header for detection messages.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_frame"

        # Convert the frame from BGR to RGB as required by YOLO.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection using the YOLO model.
        results = self.model.predict(
            rgb_frame,
            device=self.device,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Parse detection results.
        raw_detections = parse_detection_result(
            results[0], TARGET_CLASS_IDS, CUSTOM_LABELS, self.model
        )
        # raw_detections is a list of tuples:
        # (class_id, conf, x1, y1, x2, y2, display_label)

        # ---------------------------
        # Object Tracking Logic
        # ---------------------------
        current_detections = []
        for (class_id, conf, x1, y1, x2, y2, display_label) in raw_detections:
            track_id = self.assign_track_id(x1, y1, x2, y2)

            # If the track exists, optionally smooth the bounding box.
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

        # Increase miss_count for tracks not detected in the current frame.
        for t_id in list(self.track_history.keys()):
            if t_id not in current_detections:
                self.track_history[t_id]["miss_count"] += 1
                if self.track_history[t_id]["miss_count"] > MAX_MISSES:
                    del self.track_history[t_id]

        # ---------------------------
        # Build and Publish Detection2DArray Message
        # ---------------------------
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = header

        for t_id, info in self.track_history.items():
            if info["miss_count"] <= MAX_MISSES:
                x1, y1, x2, y2 = info["bbox"]
                conf = info["conf"]
                label = info["label"]

                # Compute bounding box center and size.
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                detection_msg = Detection2D()
                detection_msg.header = header

                # Populate bounding box.
                bbox = BoundingBox2D()
                bbox.center = Pose2D(x=cx, y=cy, theta=0.0)
                bbox.size_x = float(width)
                bbox.size_y = float(height)
                detection_msg.bbox = bbox

                # Populate detection hypothesis.
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = f"{t_id}:{label}"
                hypothesis.score = conf
                detection_msg.results.append(hypothesis)

                detection_array_msg.detections.append(detection_msg)

        self.publisher_.publish(detection_array_msg)
        self.get_logger().info(f"Published {len(detection_array_msg.detections)} detections.")

        # Optionally publish the frame with detection overlays.
        if PUBLISH_FRAMES_WITH_DETECTIONS:
            # Copy frame to overlay detections.
            detection_frame = frame.copy()
            for t_id, info in self.track_history.items():
                if info["miss_count"] <= MAX_MISSES:
                    x1, y1, x2, y2 = info["bbox"]
                    label = info["label"]
                    conf = info["conf"]
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(detection_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Convert the detection frame to a ROS Image message.
            img_msg = self.bridge.cv2_to_imgmsg(detection_frame, encoding="bgr8")
            img_msg.header = header
            self.detection_frame_pub.publish(img_msg)
            self.get_logger().info("Published detection frame.")

    def assign_track_id(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Assign an existing track ID to a detection (using centroid distance) or
        create a new track if no match is found.
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
        Generate a unique track ID.
        """
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def destroy_node(self):
        """
        Release the video capture resource before shutting down.
        """
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_name, device)
    if not hasattr(model, 'device'):
        model.device = device
    node = CombinedDetectionNode(model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Combined Detection Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()