"""
Camera Node Module

This module implements a ROS2 node that publishes camera frames as ROS Image messages.
It can handle both live camera feed and video file playback, publishing frames at 10Hz.

Features:
- Supports both webcam and MP4 video file input
- Automatic video looping when reaching end of file
- Converts OpenCV frames to ROS Image messages
- Publishes images with current ROS time stamps
"""

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Configuration for video source
USE_MP4_FILE = False  # Set to True to use run.mp4, False for live camera
video_path = "software/src/videos/football.mp4"

class CameraNode(Node):
    """
    A ROS2 node that captures frames from a camera or video file and publishes them.
    
    This node can operate in two modes:
    1. Camera mode: Captures frames from a connected webcam
    2. Video file mode: Reads frames from an MP4 file with automatic looping
    
    Publishers:
        /camera/image (sensor_msgs/Image): Publishes captured frames as ROS Image messages
    """

    def __init__(self):
        """
        Initialize the camera node with necessary publishers and video capture setup.
        
        The constructor:
        - Sets up the ROS2 publisher for image messages
        - Initializes video capture (either camera or file)
        - Creates a timer for periodic frame publishing
        - Sets up CvBridge for OpenCV-ROS image conversion
        """
        super().__init__('camera_node')

        ## A publisher to publish on topic /camera/image
        self.publisher_ = self.create_publisher(Image, '/camera/image', 10)

        # Create a timer to periodically read frames from the camera/video.
        timer_period = 0.1  # seconds (10 Hz frame rate)

        ## Camera node timer
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Depending on the toggle, open the appropriate video source.
        if USE_MP4_FILE:
            # Use local MP4 file
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.get_logger().error("Could not open video file.")
            else:
                self.get_logger().info("Opened video file for video streaming.")
        else:
            # Use RTSP feed from IP camera
            rtsp_url = "rtsp://192.168.145.25:8554/main.264"  # <-- Adjust this as needed
            self.cap = cv2.VideoCapture(rtsp_url)
            if not self.cap.isOpened():
                self.get_logger().error(f"Could not open RTSP stream at {rtsp_url}.")
            else:
                self.get_logger().info(f"Opened RTSP stream from {rtsp_url}.")

        ## CvBridge instance
        self.bridge = CvBridge()
        self.get_logger().info("Camera node started, publishing to /camera/image...")

    def timer_callback(self):
        """
        Timer callback function executed at 10Hz to publish camera frames.
        
        This method:
        1. Captures a new frame from the video source
        2. Handles video file looping if end-of-file is reached
        3. Converts the OpenCV frame to a ROS Image message
        4. Adds current timestamp to the message
        5. Publishes the frame to /camera/image topic
        
        Error handling is implemented for failed frame captures and EOF conditions.
        """
        ret, frame = self.cap.read()

        # Handle video file looping
        if USE_MP4_FILE and not ret:
            self.get_logger().warning("Reached end of video. Looping back to start.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to first frame
            ret, frame = self.cap.read()

        # Error handling for frame capture failures
        if not ret:
            self.get_logger().warning("Failed to read frame from camera/video source.")
            return

        # Convert and publish the frame
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()  # Add timestamp
        self.publisher_.publish(ros_image)

    def destroy_node(self):
        """
        Clean up node resources before shutdown.
        
        Ensures proper release of video capture resources to prevent device/file handle leaks.
        """
        # Release the video/camera resource before shutting down
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main():
    """
    Main entry point for the camera node.
    
    Initializes ROS2 context, creates and runs the camera node.
    Handles graceful shutdown on keyboard interrupt.
    """
    rclpy.init(args=sys.argv)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info("Camera node stopped by user.")
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
