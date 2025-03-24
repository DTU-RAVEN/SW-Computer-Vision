#!/usr/bin/env python3
"""
A ROS2 node that manages autonomous payload dropping for UAV missions.

This node implements a state machine that coordinates payload dropping based on multiple conditions:
- Altitude threshold verification (minimum 50' AGL)
- Geographic boundary containment check
- Object detection integration
- Lap-based dropping logic to prevent multiple drops per lap

Publishers:
    - /payload_release (Bool): Triggers the physical payload release mechanism
    
Subscribers:
    - /telemetry/gps (NavSatFix): Current GPS position
    - /telemetry/altitude (Float32): Current altitude AGL in feet
    - /vision/object_spotted (String): Object detection results in JSON format
    - /mission/lap_completed (Bool): Signals completion of mission lap
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Float32, Bool
import json
import math

# PARAMETERS (modify as needed)
CONFIDENCE_THRESHOLD = 0.5         # Confidence threshold for detection
CONSECUTIVE_FRAMES_REQUIRED = 3    # Number of consecutive frames required with a valid detection

class PayloadDropNode(Node):
    def __init__(self):
        super().__init__('payload_drop_node')

        # ----------------
        # PARAMETERS
        # ----------------
        self.min_drop_altitude = 50.0  # feet AGL
        # Define the boundary polygon of the air drop area 
        # (in GPS lat/lon as a list of (lat, lon) tuples).
        self.drop_boundary_polygon = [
            (38.315386, -76.550875),
            (38.315683, -76.552586),
            (38.315895, -76.552519),
            (38.315607, -76.550800)
        ]
        # Track whether a payload has already been dropped this lap
        self.payload_dropped_this_lap = False

        # New internal state: count consecutive frames with a valid detection
        self.consecutive_detections = 0

        # Subscriptions
        self.gps_subscription = self.create_subscription(
            NavSatFix,
            '/telemetry/gps',
            self.gps_callback,
            10
        )
        self.alt_subscription = self.create_subscription(
            Float32,
            '/telemetry/altitude',
            self.alt_callback,
            10
        )
        self.object_subscription = self.create_subscription(
            String,
            '/vision/object_spotted',
            self.object_detection_callback,
            10
        )
        self.lap_subscription = self.create_subscription(
            Bool,
            '/mission/lap_completed',
            self.lap_completed_callback,
            10
        )

        # Publishers
        self.drop_publisher = self.create_publisher(
            Bool,
            '/payload_release',
            10
        )

        # Internal state variables for telemetry
        self.current_lat = None
        self.current_lon = None
        self.current_alt_ft = 0.0  # altitude in feet AGL

        self.get_logger().info("PayloadDropNode initialized and running...")

    # ---------------------------
    # CALLBACKS
    # ---------------------------
    def gps_callback(self, msg: NavSatFix):
        """ Store the most recent GPS position. """
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def alt_callback(self, msg: Float32):
        """ Store the most recent altitude (in feet, if your system provides it). """
        self.current_alt_ft = msg.data

    def object_detection_callback(self, msg: String):
        """
        Processes incoming object detection messages and initiates payload drop if conditions are met.
        
        The detection message is expected to be a JSON string containing a 'detections' array.
        Each detection should include at least a confidence value under the key "conf".
        
        Note:
            Instead of dropping on any detection, this version requires that a valid detection 
            (one with confidence >= CONFIDENCE_THRESHOLD) be observed in consecutive frames.
        """
        if not self.is_ready_for_drop():
            return

        data = json.loads(msg.data)
        detections = data.get("detections", [])

        valid_detection = False
        for detection in detections:
            # Check if the detection meets the confidence threshold
            if detection.get("conf", 0) >= CONFIDENCE_THRESHOLD:
                valid_detection = True
                break

        if valid_detection:
            self.consecutive_detections += 1
            self.get_logger().info(f"Valid detection count: {self.consecutive_detections}")
            if self.consecutive_detections >= CONSECUTIVE_FRAMES_REQUIRED:
                self.get_logger().info("Required consecutive detections reached. Triggering drop.")
                self.trigger_payload_drop()
                self.consecutive_detections = 0  # Reset counter after drop
        else:
            self.consecutive_detections = 0

    def lap_completed_callback(self, msg: Bool):
        """
        Resets drop eligibility when a new lap is completed.
        """
        if msg.data:
            self.get_logger().info("Lap completed, resetting payload drop availability.")
            self.payload_dropped_this_lap = False

    # ---------------------------
    # SUPPORT FUNCTIONS
    # ---------------------------
    def is_ready_for_drop(self) -> bool:
        """
        Checks if the drone meets all conditions for a safe payload drop:
        1. Has not already dropped a payload this lap.
        2. Has valid GPS data.
        3. Is above the minimum safe altitude.
        4. Is within the designated drop boundary.
        """
        if self.payload_dropped_this_lap:
            return False
        if self.current_lat is None or self.current_lon is None:
            return False
        if self.current_alt_ft < self.min_drop_altitude:
            return False
        if not self.is_within_boundary(self.current_lat, self.current_lon):
            return False
        return True

    def trigger_payload_drop(self):
        """ Publishes a drop command to /payload_release and marks the payload as dropped. """
        drop_msg = Bool()
        drop_msg.data = True
        self.drop_publisher.publish(drop_msg)
        self.payload_dropped_this_lap = True
        self.get_logger().info("Payload drop triggered!")

    def is_within_boundary(self, lat: float, lon: float) -> bool:
        """
        Uses a simplified ray-casting algorithm to determine if a point is inside a polygon.
        (No modifications to coordinate handling are made.)
        """
        polygon = self.drop_boundary_polygon
        inside = False
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            lat_i, lon_i = polygon[i]
            lat_j, lon_j = polygon[j]

            cond_y = ((lon_i > lon) != (lon_j > lon))
            if cond_y:
                x_intersect = (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
                if x_intersect > lat:
                    inside = not inside
        return inside

def main(args=None):
    rclpy.init(args=args)
    node = PayloadDropNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("PayloadDropNode stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
