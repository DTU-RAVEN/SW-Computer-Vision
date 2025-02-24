#!/usr/bin/env python3
import threading
import cv2
from flask import Flask, Response

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import the detection message type.
from vision_msgs.msg import Detection2DArray

# Global variables to hold the latest annotated frame and detections.
outputFrame = None
latestDetections = None
lock = threading.Lock()

# Initialize Flask app for streaming.
app = Flask(__name__)

@app.route("/video_feed")
def video_feed():
    """
    Flask route to serve the MJPEG video stream.
    Open this URL on your Mac: http://<jetson_ip>:5001/video_feed
    """
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_stream():
    """
    Generator function that continuously encodes the latest frame as JPEG
    and yields it for the MJPEG stream.
    """
    global outputFrame
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
            frame = encodedImage.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

class StreamDetectionsNode(Node):
    """
    ROS2 node that subscribes to raw camera images and detection results.
    It overlays the detection bounding boxes (from /vision/object_spotted)
    onto the camera image, and then updates a global frame for streaming.
    """
    def __init__(self):
        super().__init__('stream_detections_node')
        self.bridge = CvBridge()
        self.get_logger().info("Stream Detections Node started.")
        
        # Subscribe to raw camera images.
        self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        # Subscribe to detection results from the object detection node.
        self.create_subscription(
            Detection2DArray,
            '/vision/object_spotted',
            self.detection_callback,
            10
        )
        
        self.get_logger().info("Subscribed to /camera/image and /vision/object_spotted.")

    def detection_callback(self, msg):
        """
        Callback for detection messages.
        Simply store the latest Detection2DArray message.
        """
        global latestDetections
        with lock:
            latestDetections = msg

    def image_callback(self, msg):
        """
        Callback for incoming camera images.
        Converts the image to an OpenCV frame and, if detection data is available,
        overlays the bounding boxes and labels onto the frame.
        """
        global outputFrame, latestDetections
        try:
            # Convert ROS Image to OpenCV BGR image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error("Error converting image: " + str(e))
            return
        
        with lock:
            # If detection results are available, overlay them.
            if latestDetections is not None:
                for detection in latestDetections.detections:
                    # Each detection contains a bounding box in Detection2D.bbox.
                    # The bbox has a center (Pose2D) and size (size_x, size_y).
                    cx = detection.bbox.center.x
                    cy = detection.bbox.center.y
                    width = detection.bbox.size_x
                    height = detection.bbox.size_y
                    x1 = int(cx - width / 2)
                    y1 = int(cy - height / 2)
                    x2 = int(cx + width / 2)
                    y2 = int(cy + height / 2)
                    
                    # Extract label and score from the first hypothesis, if available.
                    if detection.results:
                        label = detection.results[0].id
                        score = detection.results[0].score
                    else:
                        label = "N/A"
                        score = 0.0
                        
                    # Draw the bounding box.
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw the label and score above the box.
                    text = f"{label}: {score:.2f}"
                    cv2.putText(cv_image, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update the global frame with the annotated image.
            outputFrame = cv_image.copy()

def start_flask():
    """
    Start the Flask server for streaming.
    """
    app.run(host="0.0.0.0", port=5004, threaded=True)

def main(args=None):
    rclpy.init(args=args)
    node = StreamDetectionsNode()
    
    # Start the Flask server in a separate thread.
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()