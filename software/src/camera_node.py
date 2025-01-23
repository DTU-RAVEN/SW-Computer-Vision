import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Toggle between using live camera feed and MP4 file.
USE_MP4_FILE = True  # Set to True to use run.mp4, False for live camera
video_path = "software/src/videos/football.mp4"

class CameraNode(Node):
    """ @brief Camera node class
    """
    def __init__(self):
        """ @brief Camera Node constructor function
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
            ## Video capture instance
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.get_logger().error("Could not open video file run.mp4.")
            else:
                self.get_logger().info("Opened run.mp4 for video streaming.")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error("Could not open camera.")

        ## CvBridge instance
        self.bridge = CvBridge()
        self.get_logger().info("Camera node started, publishing to /camera/image...")

    def timer_callback(self):
        """ @brief Camera node timer callback
        """
        ret, frame = self.cap.read()

        # If using MP4 and we reach end-of-file, loop back.
        if USE_MP4_FILE and not ret:
            self.get_logger().warning("Reached end of video. Looping back to start.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        # If reading frame failed (for camera or unexpected mp4 issue), log and skip publishing.
        if not ret:
            self.get_logger().warning("Failed to read frame from camera/video source.")
            return

        # Convert OpenCV image (BGR) to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Attach the current ROS2 time to the header
        ros_image.header.stamp = self.get_clock().now().to_msg()

        # Publish the image
        self.publisher_.publish(ros_image)

    def destroy_node(self):
        """ @brief Destroy this node
        """
        # Release the video/camera resource before shutting down
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main():
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
