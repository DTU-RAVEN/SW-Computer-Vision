#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import json

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        # Subscribe to raw camera feed
        self.subscription_image = self.create_subscription(
            Image,
            '/camera/image',
            self.camera_callback,
            10
        )

        # Subscribe to detection results (JSON-encoded bounding boxes, etc.)
        self.subscription_detections = self.create_subscription(
            String,
            '/vision/object_spotted',
            self.detections_callback,
            10
        )

        self.bridge = CvBridge()
        self.current_detections = []  # Will hold bounding boxes from last detection message

        # Create a small timer to periodically allow OpenCV to process GUI events
        self.timer = self.create_timer(0.05, self.opencv_gui_loop)

    def camera_callback(self, msg):
        """Callback for receiving camera frames (ROS Image)."""
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
        """Callback for receiving detection results (JSON)."""
        try:
            data = json.loads(msg.data)
            self.current_detections = data.get("detections", [])
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            return

    def opencv_gui_loop(self):
        """
        This timer callback just calls `cv2.waitKey(1)` so that OpenCV
        processes the GUI events. Without it, the imshow window won’t update.
        """
        cv2.waitKey(1)

def main(args=None):
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
