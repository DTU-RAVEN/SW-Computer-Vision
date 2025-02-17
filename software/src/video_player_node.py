import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import message_filters

class VisualizationNode(Node):
    """
    A ROS2 node for visualizing object detection results overlaid on camera images.
    
    This node subscribes to:
      1. Camera images ('/camera/image')
      2. Detection results ('/vision/object_spotted') published as a Detection2DArray
     
    It synchronizes these topics based on header timestamps using message_filters'
    ApproximateTimeSynchronizer, ensuring that each displayed frame is annotated with
    its corresponding detections.
    """
    def __init__(self):
        super().__init__('visualization_node')
        
        # Initialize CvBridge for converting ROS images to OpenCV images
        self.bridge = CvBridge()
        
        # Create message_filters subscribers for the image and detection topics
        image_sub = message_filters.Subscriber(self, Image, '/camera/image')
        detection_sub = message_filters.Subscriber(self, Detection2DArray, '/vision/object_spotted')
        
        # ApproximateTimeSynchronizer parameters:
        #   - queue_size: maximum number of messages to store for synchronization
        #   - slop: allowable time difference (in seconds) between messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detection_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        
    def synced_callback(self, image_msg, detections_msg):
        """
        Callback triggered when an image and detection message are available within the specified time window.
        
        Args:
            image_msg (sensor_msgs.msg.Image): The camera image message.
            detections_msg (vision_msgs.msg.Detection2DArray): The detection results.
        """
        try:
            # Convert the ROS Image to an OpenCV image (BGR)
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Iterate through each Detection2D in the array and overlay the bounding boxes
        for detection in detections_msg.detections:
            # Extract bounding box information from BoundingBox2D
            cx = detection.bbox.center.x
            cy = detection.bbox.center.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

            # Extract hypothesis details (label and confidence)
            if detection.results:
                hypothesis = detection.results[0]
                label = hypothesis.hypothesis
                confidence = hypothesis.score
            else:
                label = "Unknown"
                confidence = 0.0

            # Convert center-based coordinates to top-left and bottom-right corners
            x1 = int(cx - width / 2)
            y1 = int(cy - height / 2)
            x2 = int(cx + width / 2)
            y2 = int(cy + height / 2)

            # Draw the rectangle and label on the image
            color = (0, 255, 0)  # Green for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the annotated frame
        cv2.imshow("Synchronized Detections", frame)
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
        cv2.destroyAllWindows()  # Ensure that OpenCV windows are closed
        rclpy.shutdown()

if __name__ == '__main__':
    main()