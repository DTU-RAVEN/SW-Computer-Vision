#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Float32, Bool
import json
import math

class PayloadDropNode(Node):
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
        # Could use a library like shapely, but for simplicity, let's do our own point-in-polygon check.

        # We track whether we have already dropped a payload this lap
        self.payload_dropped_this_lap = False

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

        # Internal state variables
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
        Each detection can represent various objects (e.g., cars, persons, targets).
        
        Args:
            msg (String): JSON formatted string containing detection results
                Format: {"detections": [{object data...}, ...]}
        
        Note:
            The current implementation drops on any detection. In practice, you might
            want to filter based on specific object types or confidence scores.
        """
        if not self.is_ready_for_drop():
            return

        # If we want to do logic about *what* object is detected:
        data = json.loads(msg.data)
        detections = data.get("detections", [])

        # For demonstration: if we detect at least one relevant object, we drop.
        if len(detections) > 0:
            self.get_logger().info(f"Objects detected: {len(detections)}. Checking drop conditions.")
            # Additional check: Make sure the object is actually a target we want. 
            # For example, if we want Car or Person, etc. 
            # In this example, we'll just drop if anything is detected.
            self.trigger_payload_drop()

    def lap_completed_callback(self, msg: Bool):
        """
        This topic indicates the UAV has completed another waypoint lap.
        When True, we reset our drop eligibility.
        """
        if msg.data:
            self.get_logger().info("Lap completed, resetting payload drop availability.")
            self.payload_dropped_this_lap = False

    # ---------------------------
    # SUPPORT FUNCTIONS
    # ---------------------------
    def is_ready_for_drop(self) -> bool:
        """
        Validates all conditions required for a safe payload drop.
        
        This method implements a multi-stage validation process:
        1. Verifies lap-based drop eligibility (one drop per lap)
        2. Confirms valid GPS signal
        3. Ensures minimum safe altitude
        4. Validates position within designated drop zone
        
        Returns:
            bool: True if all drop conditions are satisfied, False otherwise
        """
        # Already dropped a payload this lap?
        if self.payload_dropped_this_lap:
            return False

        # Must have valid GPS
        if self.current_lat is None or self.current_lon is None:
            return False

        # Check altitude
        if self.current_alt_ft < self.min_drop_altitude:
            return False

        # Check if inside boundary
        if not self.is_within_boundary(self.current_lat, self.current_lon):
            return False

        return True

    def trigger_payload_drop(self):
        """ Publishes a drop command to /payload_release. """
        drop_msg = Bool()
        drop_msg.data = True
        self.drop_publisher.publish(drop_msg)
        self.payload_dropped_this_lap = True
        self.get_logger().info("Payload drop triggered!")

    def is_within_boundary(self, lat: float, lon: float) -> bool:
        """
        Implements the ray-casting algorithm to determine if a point lies within a polygon.
        
        This implementation uses a simplified planar geometry approach suitable for
        small geographic areas. For more precise results or larger areas, consider
        using a geodesic library like GeoPy or Shapely.
        
        Algorithm:
        1. Casts a ray from the test point
        2. Counts intersections with polygon edges
        3. Uses even-odd rule to determine containment
        
        Args:
            lat (float): Latitude of the point to test
            lon (float): Longitude of the point to test
            
        Returns:
            bool: True if point is inside the polygon, False otherwise
            
        """
        polygon = self.drop_boundary_polygon
        inside = False
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            lat_i, lon_i = polygon[i]
            lat_j, lon_j = polygon[j]

            # Check if the point is between the lat_i, lat_j band
            cond_y = ((lon_i > lon) != (lon_j > lon))
            # Find where the line crosses the x-lat if we draw a horizontal line at 'lon'
            if cond_y:
                x_intersect = (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
                if x_intersect > lat:  # Flip the comparison if you prefer reversing lat/lon
                    inside = not inside
        return inside


def main(args=None):
    """
    Main entry point for the payload drop node.
    
    Initializes the ROS2 system, creates the node instance, and handles
    graceful shutdown on keyboard interrupt.
    
    Args:
        args: Command line arguments passed to rclpy.init()
    """
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
