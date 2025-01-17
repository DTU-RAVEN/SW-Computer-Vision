#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Float32, Bool
import json
import math

class PayloadDropNode(Node):
    """
    This ROS2 node coordinates payload drop logic:
    1) Checks if UAV is above 50' altitude.
    2) Checks if UAV is within the air drop boundary polygon.
    3) Subscribes to object detection messages.
    4) Ensures a drop happens only if a new lap has started.
    5) Publishes a signal to /payload_release when conditions are met.
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
        Parse object detection results. If the conditions to drop are satisfied,
        call a function to trigger the payload release.
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
        Checks the main conditions for dropping a payload:
        1. We have not dropped a payload this lap.
        2. We are inside the air drop boundary polygon.
        3. The altitude is >= minimum drop altitude.
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
        Point-in-polygon check using the ray-casting algorithm.
        Returns True if (lat, lon) is inside the polygon.
        Polygon is a list of (lat, lon) tuples.
        """
        # Basic ray-casting or winding number approach in lat-lon.  
        # For small areas, you can treat lat/lon as planar coordinates.  
        # For large or more accurate checks, use a geospatial library.
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
