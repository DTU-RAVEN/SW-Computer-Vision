#!/usr/bin/env python3
"""
This script runs a ROS2 node that records detection frames received on the
/vision/detection_frames topic and saves them to video files. It maintains the
CONTINUE_SAVING_TO_NEW_FILES functionality to continue recording to new files
when the recording duration elapses.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

CONTINUE_SAVING_TO_NEW_FILES = False

class DetectionFrameRecordingNode(Node):
    def __init__(self):
        super().__init__('detection_frame_recording_node')
        # Subscribe to the detection frames topic.
        self.subscription = self.create_subscription(
            Image,
            '/vision/detection_frames',
            self.frame_callback,
            10
        )
        self.bridge = CvBridge()

        self.video_writer = None
        self.frame_width = None
        self.frame_height = None

        self.start_time = time.time()
        self.duration = 40.0   # Record for 40 seconds.
        self.VIDEO_FPS = 10.0  # Expected frame rate.
        self.file_index = 1    # New file counter

    def frame_callback(self, image_msg):
        current_time = time.time()
        # Check if current file duration has been exceeded.
        if self.video_writer is not None and (current_time - self.start_time > self.duration):
            if CONTINUE_SAVING_TO_NEW_FILES:
                self.get_logger().info(
                    f"Recording duration reached. Finishing file detection_frames_{self.file_index}.mp4 and starting new file."
                )
                self.video_writer.release()
                self.video_writer = None
                self.file_index += 1
                self.start_time = current_time
            else:
                self.get_logger().info("Recording duration reached. Shutting down.")
                if self.video_writer is not None:
                    self.video_writer.release()
                self.destroy_node()
                rclpy.shutdown()
                return

        try:
            # Convert ROS Image message to an OpenCV BGR image.
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Detection frames already have overlayed bounding boxes.
        annotated_frame = frame

        # Initialize the video writer if not already initialized.
        if self.video_writer is None:
            self.frame_height, self.frame_width = annotated_frame.shape[:2]
            filename = (f"detections_{self.file_index}.mp4"
                        if CONTINUE_SAVING_TO_NEW_FILES else "detections.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, self.VIDEO_FPS,
                (self.frame_width, self.frame_height)
            )
            if not self.video_writer.isOpened():
                self.get_logger().error("VideoWriter failed to open!")
            else:
                self.get_logger().info(f"Video recording started: {filename}")
                
        self.video_writer.write(annotated_frame)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionFrameRecordingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recording stopped by user.")
    finally:
        if node.video_writer is not None:
            node.video_writer.release()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()