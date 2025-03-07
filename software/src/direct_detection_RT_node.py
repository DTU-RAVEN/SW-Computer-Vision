#!/usr/bin/env python3
"""
Camera and Detection Node using TensorRT

This node:
- Opens a video source (either an RTSP stream or a local MP4 file).
- Reads frames directly (at 10Hz) without publishing them on a ROS topic.
- Performs object detection and simple tracking on each frame using a YOLOv8 TensorRT engine.
- Publishes the detection results as a Detection2DArray message on the topic '/vision/object_spotted'.

Configuration:
    - USE_MP4_FILE: Toggle between using a video file or an RTSP stream.
    - video_path: Path to the MP4 file (if USE_MP4_FILE is True).
    - rtsp_url: URL of the RTSP stream (if USE_MP4_FILE is False).
    - The engine file "yolov8s.engine" should be generated from the YOLOv8 model.
"""

import sys
import cv2
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, BoundingBox2D, Detection2DArray

# TensorRT and PyCUDA imports.
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes the CUDA driver.

# ---------------------------
# GLOBAL CONFIGURATION
# ---------------------------
USE_MP4_FILE = False  # Set to True to use video file input; False to use RTSP stream.
video_path = "software/src/videos/football.mp4"
rtsp_url = "rtsp://192.168.145.25:8554/main.264"  # Adjust as needed.

# ---------------------------
# DETECTION AND TRACKING PARAMETERS
# ---------------------------
# In this TensorRT version, the engine file is used instead of a .pt model.
model_engine_path = "yolov8s.engine"
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

# ---------------------------
# TENSORRT INFERENCE HELPER CLASS
# ---------------------------
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path, runtime):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

class TRTInference:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = load_engine(engine_path, self.runtime)
        self.context = self.engine.create_execution_context()
        # For simplicity, assume the engine has two bindings: binding 0 is input, binding 1 is output.
        self.input_binding_idx = 0
        self.output_binding_idx = 1

        # Get input properties.
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        # e.g., expected shape is [batch, channels, height, width]
        self.input_size = trt.volume(self.input_shape)
        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_binding_idx))

        # Get output properties.
        self.output_shape = self.engine.get_binding_shape(self.output_binding_idx)
        self.output_size = trt.volume(self.output_shape)
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_binding_idx))

        # Allocate device memory for input and output.
        self.d_input = cuda.mem_alloc(self.input_size * np.dtype(self.input_dtype).itemsize)
        self.d_output = cuda.mem_alloc(self.output_size * np.dtype(self.output_dtype).itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def infer(self, image):
        """
        Preprocesses the image, executes inference, and returns the raw output.
        Assumes the engine expects images in RGB, normalized to [0,1] with shape [batch, channels, height, width].
        """
        # Retrieve expected input dimensions.
        _, channels, height, width = self.input_shape

        # Resize image to expected size.
        resized = cv2.resize(image, (width, height))
        # Convert BGR (from OpenCV) to RGB.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize image to [0,1].
        rgb = rgb.astype(self.input_dtype) / 255.0
        # Change from HWC to CHW format.
        chw = np.transpose(rgb, (2, 0, 1))
        # Add batch dimension and flatten.
        input_data = np.expand_dims(chw, axis=0).ravel()

        # Transfer input data to GPU.
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Allocate output buffer and copy result back from GPU.
        output = np.empty(self.output_size, dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        # Reshape output to the expected dimensions.
        output = output.reshape(self.output_shape)
        return output

# ---------------------------
# DETECTION RESULT PARSING FUNCTION
# ---------------------------
def parse_detection_result_trt(raw_output, target_class_ids, custom_labels, conf_threshold):
    """
    Parses the raw TensorRT output.
    Assumes raw_output is a 2D array of shape (N, 6) where each row contains:
      [x1, y1, x2, y2, confidence, class_id]
    Returns a list of tuples:
      (class_id, conf, x1, y1, x2, y2, display_label)
    """
    detections = []
    for det in raw_output:
        conf = det[4]
        if conf < conf_threshold:
            continue
        class_id = int(det[5])
        if class_id not in target_class_ids:
            continue
        x1, y1, x2, y2 = map(int, det[:4])
        display_label = custom_labels.get(class_id, "Unknown")
        detections.append((class_id, conf, x1, y1, x2, y2, display_label))
    return detections

# ---------------------------
# COMBINED CAMERA & DETECTION NODE
# ---------------------------
class CombinedDetectionNode(Node):
    """
    A combined ROS2 node that performs both camera frame acquisition and object detection
    using a TensorRT engine.
    """
    def __init__(self):
        super().__init__('combined_detection_node')

        # Publisher for Detection2DArray messages.
        self.publisher_ = self.create_publisher(Detection2DArray, '/vision/object_spotted', 10)

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
            self.cap = cv2.VideoCapture(rtsp_url)
            if not self.cap.isOpened():
                self.get_logger().error(f"Could not open RTSP stream at {rtsp_url}.")
            else:
                self.get_logger().info(f"Opened RTSP stream from {rtsp_url}.")

        # Initialize TensorRT model.
        self.engine_path = model_engine_path  # path to the .engine file
        self.trt_infer = TRTInference(self.engine_path)
        self.get_logger().info("TensorRT engine loaded and ready.")

        # Tracking state.
        self.track_history = {}
        self.next_track_id = 0

    def timer_callback(self):
        """
        Timer callback that:
          1. Captures a frame from the camera/video.
          2. Runs TensorRT inference and performs detection parsing.
          3. Applies simple tracking.
          4. Publishes the detection results.
        """
        ret, frame = self.cap.read()

        # If using video file, loop the video upon reaching its end.
        if USE_MP4_FILE and not ret:
            self.get_logger().warning("Reached end of video. Looping back to start.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning("Failed to read frame from video source.")
            return

        # Create a header for detection messages.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_frame"

        # Run inference using TensorRT.
        start_time = time.time()
        raw_output = self.trt_infer.infer(frame)
        inference_time = (time.time() - start_time) * 1000  # in milliseconds
        self.get_logger().info(f"Inference time: {inference_time:.1f} ms")

        # Parse detection results.
        raw_detections = parse_detection_result_trt(
            raw_output, TARGET_CLASS_IDS, CUSTOM_LABELS, conf_threshold
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
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CombinedDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Combined Detection Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()