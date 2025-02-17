import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import time
import message_filters

class VideoRecordingNode(Node):
    def __init__(self):
        super().__init__('video_recording_node')
        # Create message_filters subscribers for image and detection topics.
        image_sub = message_filters.Subscriber(self, Image, '/camera/image')
        detection_sub = message_filters.Subscriber(self, Detection2DArray, '/vision/object_spotted')
        # Use an ApproximateTimeSynchronizer to pair images with detections.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detection_sub],
            queue_size=20,
            slop=0.2  # allow up to 200ms difference between image and detection
        )
        self.ts.registerCallback(self.synced_callback)

        self.bridge = CvBridge()

        self.video_writer = None
        self.frame_width = None
        self.frame_height = None

        self.start_time = time.time()
        self.duration = 20.0   # Record for 20 seconds.
        self.VIDEO_FPS = 10.0  # Use camera's 10 Hz rate for both capture and playback.

    def synced_callback(self, image_msg, detection_msg):
        current_time = time.time()
        # Stop recording if 20 seconds have passed.
        if current_time - self.start_time > self.duration:
            self.get_logger().info("Recording duration reached. Shutting down.")
            if self.video_writer is not None:
                self.video_writer.release()
                self.get_logger().info("Released video writer.")
            self.destroy_node()
            rclpy.shutdown()
            return

        try:
            # Convert ROS Image to OpenCV BGR image.
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        annotated_frame = frame.copy()

        # Overlay detections from the synchronized Detection2DArray.
        for detection in detection_msg.detections:
            # Compute top-left and bottom-right coordinates from center-based bounding box.
            cx = detection.bbox.center.x
            cy = detection.bbox.center.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            x1 = int(cx - width / 2)
            y1 = int(cy - height / 2)
            x2 = int(cx + width / 2)
            y2 = int(cy + height / 2)

            # In ROS Foxy, ObjectHypothesisWithPose has only 'id' (a string) and 'score'.
            if detection.results:
                hypothesis = detection.results[0]
                # Our detection node stores track_id and label in a composite string like "3:person".
                combined_str = hypothesis.id
                if ':' in combined_str:
                    _, label = combined_str.split(':', 1)
                else:
                    label = combined_str
                conf = hypothesis.score
            else:
                label = "Unknown"
                conf = 0.0

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Initialize video writer if not already.
        if self.video_writer is None:
            self.frame_height, self.frame_width = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter('detections.mp4', fourcc,
                                                self.VIDEO_FPS,
                                                (self.frame_width, self.frame_height))
            if not self.video_writer.isOpened():
                self.get_logger().error("VideoWriter failed to open!")
            else:
                self.get_logger().info("Video recording started: detections.mp4")
                
        self.video_writer.write(annotated_frame)

def main(args=None):
    rclpy.init(args=args)
    node = VideoRecordingNode()
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