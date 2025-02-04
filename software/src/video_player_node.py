#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import json

class VisualizationNode(Node):
    """
    A ROS2 node for visualizing object detection results overlaid on camera images.
    
    This node subscribes to two topics:
    1. Camera images ('/camera/image')
    2. Detection results ('/vision/object_spotted')
    
    It processes incoming detection data (bounding boxes, labels, confidence scores)
    and visualizes them in real-time on the camera feed using OpenCV.
    """
    def __init__(self):
        """
        Initialize the visualization node with necessary subscribers and data structures.
        
        Sets up:
        - Image subscriber for raw camera feed
        - String subscriber for JSON-formatted detection results
        - CvBridge for ROS<->OpenCV image conversion
        - Timer for OpenCV GUI updates
        - Storage for current detection results
        """
        super().__init__('visualization_node')

        ## Subscribe to raw camera feed
        self.subscription_image = self.create_subscription(
            Image,
            '/camera/image',

            ## Camera callback
            self.camera_callback,
            10
        )

        # Subscribe to detection results (JSON-encoded bounding boxes, etc.)
        self.subscription_detections = self.create_subscription(
            String,
            '/vision/object_spotted',

            ## Detection callback
            self.detections_callback,
            10
        )

        ## CvBridge instance
        self.bridge = CvBridge()

        ## Will hold bounding boxes from last detection message
        self.current_detections = []

        ## Create a small timer to periodically allow OpenCV to process GUI events
        self.timer = self.create_timer(0.05, self.opencv_gui_loop)

    def camera_callback(self, msg):
        """
        Process incoming camera frames and overlay detection visualizations.
        
        Args:
            msg (sensor_msgs.msg.Image): The incoming ROS Image message
        
        The method:
        1. Converts ROS Image to OpenCV format
        2. For each stored detection:
           - Extracts bounding box coordinates and metadata
           - Draws rectangle around detected object
           - Adds label with confidence score
        3. Displays the annotated frame
        """
        try:
            # Convert the ROS Image to an OpenCV image (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Draw bounding boxes for all currently stored detections
        for det in self.current_detections:
            # Each det is something like:
            # {
            #   "object_id": 0,
            #   "position": [cx, cy, width, height],
            #   "label": "...",
            #   "confidence": 0.95,
            #   "timestamp": 123456
            # }
            x_center, y_center, box_w, box_h = det["position"]
            label = det["label"]
            confidence = det["confidence"]

            # Convert center/width/height to top-left and bottom-right
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Draw the rectangle
            color = (0, 255, 0)  # green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label text above the bounding box
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Finally, show the frame in a named window
        cv2.imshow("Detections", frame)
        # We won't call cv2.waitKey(1) here—it's called in a timer callback (opencv_gui_loop).

    def detections_callback(self, msg):
        """
        Process incoming detection results from JSON format.
        
        Args:
            msg (std_msgs.msg.String): JSON-encoded detection results
        
        Expected JSON format:
        {
            "detections": [
                {
                    "object_id": int,
                    "position": [cx, cy, width, height],
                    "label": str,
                    "confidence": float,
                    "timestamp": float
                },
                ...
            ]
        }
        """
        try:
            data = json.loads(msg.data)
            self.current_detections = data.get("detections", [])
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            return

    def opencv_gui_loop(self):
        """
        Timer callback to process OpenCV GUI events.
        
        Called periodically (every 50ms) to ensure proper window updates
        and event processing in OpenCV's windowing system. Without this,
        the visualization window would appear frozen.
        """
        cv2.waitKey(1)

def main(args: list[str] = []):
    """
    Main entry point for the visualization node.
    
    Args:
        args (list[str]): Command-line arguments passed to the node
    
    Handles:
    1. ROS2 initialization
    2. Node creation and spinning
    3. Graceful shutdown with proper cleanup
    """
    rclpy.init(args=args)
    node = VisualizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()  # Close any OpenCV windows
        rclpy.shutdown()

if __name__ == '__main__':
    main()
