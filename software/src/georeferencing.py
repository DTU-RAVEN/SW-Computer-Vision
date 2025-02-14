#!/usr/bin/env python3
"""
Transform Node
===============

This node synchronizes multiple sensor topics and computes world coordinates
from pixel coordinates detected in an image. It uses data from:

  - An IMU (to obtain orientation),
  - UTM coordinates (to obtain altitude),
  - Camera intrinsic information, and
  - Pixel coordinates from image recognition.

The node performs the following operations:
  1. Retrieves the camera intrinsic matrix and computes its inverse.
  2. Computes a rotation matrix from the IMU's quaternion (using the yaw angle).
  3. Combines the rotation with the inverse intrinsic matrix to form a projection matrix.
  4. Projects pixel coordinates (converted into homogeneous form) through the camera model.
  5. Performs a ray-plane intersection assuming the ground plane is at Z = 0.
  6. Publishes the resulting world coordinates as a PoseStamped message.

Assumptions:
  - The UTM coordinates (PointStamped) provide the altitude in the z component.
  - The ground is at Z = 0 in the world frame.
  - The pixel coordinates are published as a PointStamped, with the pixel (x, y) stored in the Point field.
  - The node uses an ApproximateTimeSynchronizer to deal with small time differences between topics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion
from vision_msgs.msg import Detection2DArray
import numpy as np
import math
from typing import Optional, Tuple

class TransformNode(Node):
    """Node that transforms pixel coordinates to world coordinates using sensor data."""
    
    def __init__(self) -> None:
        """
        Initialize the TransformNode.

        Sets up subscribers for:
          - IMU data (/mavros/imu/data)
          - UTM coordinates (utm_coordinates)
          - Camera info (/camera_info)
          - Pixel coordinates (pixels)

        Also creates a publisher to output the computed world coordinates on the 'pixel_to_world' topic.
        """
        super().__init__('transform_node')
        
        # Initialize state variables.
        self.q: Optional[Quaternion] = None           # IMU orientation (quaternion)
        self.altitude: Optional[float] = None           # Altitude from UTM (assumed camera height above ground)
        self.pixels: Optional[np.ndarray] = None        # Pixel coordinates (homogeneous form)
        self.K: Optional[np.ndarray] = None             # Camera intrinsic matrix
        self.K_inv: Optional[np.ndarray] = None         # Inverse of camera intrinsic matrix
        self.P: Optional[np.ndarray] = None             # Combined projection matrix (rotation * inverse intrinsic)
        self.t: float = 0.0                             # Scaling factor used in the ray-plane intersection

        # Define QoS settings for subscribers and publishers.
        qos_profile: QoSProfile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        # Create subscribers using message_filters for synchronized callbacks.
        imu_sub: Subscriber = Subscriber(
            self, 
            Imu, 
            '/mavros/imu/data', 
            qos_profile=qos_profile
        )

        utm_sub: Subscriber = Subscriber(
            self,
            PointStamped, 
            'utm_coordinates',  
            qos_profile=qos_profile
        )

        cam_info_sub: Subscriber = Subscriber(
            self,
            CameraInfo, 
            '/camera_info', 
            qos_profile=qos_profile
        )          
        
        pixel_sub: Subscriber = Subscriber(
            self,
            Detection2DArray,
            'pixels',
            qos_profile=qos_profile
        )

        # Synchronize incoming messages with a small allowable time difference (slop).
        self.sync: ApproximateTimeSynchronizer = ApproximateTimeSynchronizer(
            [imu_sub, utm_sub, cam_info_sub, pixel_sub],
            queue_size=10, 
            slop=1.0
        )
        self.sync.registerCallback(self.callback)

        # Create a publisher to publish the computed PoseStamped (world coordinates).
        self.waypoint_pub = self.create_publisher(
            PoseStamped, 
            'pixel_to_world',
            qos_profile=qos_profile
        )

        self.get_logger().info("Created subscribers and waypoint publisher.")

    def callback(self, imu: Imu, utm: PointStamped, info: CameraInfo, detections: Detection2DArray) -> None:
        """
        Callback for synchronized messages from IMU, UTM, CameraInfo, and pixel topics.

        Extracts necessary data from the messages and triggers the computation and publishing
        of world coordinates based on the provided pixel location.
        
        Parameters:
            imu (Imu): Contains the orientation (quaternion).
            utm (PointStamped): Contains UTM coordinates; altitude is extracted from utm.point.z.
            info (CameraInfo): Provides camera intrinsic parameters.
            pixel (Detection2DArray): Contains pixel coordinates (stored in detection.bbox.center.x and detection.bbox.center.y).
        """
        # Extract altitude from UTM coordinates. (Assuming altitude is in utm.point.z)
        self.x = utm.point.x
        self.y = utm.point.y
        self.altitude = utm.point.z

         # Ensure that there is at least one detection.
        if not detections.detections:
            self.get_logger().warn("No detections received in Detection2DArray message.")
            return
        
        # Store the orientation from the IMU message.
        self.q = imu.orientation

        # Retrieve camera intrinsic parameters if not already done.
        if self.K is None:
            self.intrinsic(info)

        # Compute the projection matrix: rotation matrix (from IMU) times the inverse of the camera intrinsic matrix.
        # The type: ignore is used here because self.K_inv is Optional and we assume it is set.
        self.P = self.rotation() @ self.K_inv  # type: ignore

        # Process each detection.
        for idx, detection in enumerate(detections.detections):
            # Extract the center of the bounding box (relative to the full image).
            center = detection.bbox.center
            # Construct the homogeneous pixel coordinate vector.
            self.pixels = np.array([center.position.x, center.position.y, 1])
            # Compute the world coordinate from this pixel.
            self.publish_waypoint() # Compute the world coordinates from the pixel and publish them.
            self.get_logger().info(f"Published waypoint for detection index {idx}.")

    def rotation(self) -> np.ndarray:
        """
        Compute a 3x3 rotation matrix from the IMU quaternion.
        
        Uses the yaw angle (rotation about the Z-axis) to create a rotation matrix.
        
        Returns:
            numpy.ndarray: A 3x3 rotation matrix.
        """
        if self.q is None:
            raise ValueError("IMU orientation (self.q) is not set.")
            
        # Extract quaternion components.
        quaternion = [self.q.x, self.q.y, self.q.z, self.q.w]
        # Convert quaternion to Euler angles (roll, pitch, yaw) and use yaw.
        siny_cosp = 2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1])
        cosy_cosp = 1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Create a rotation matrix for a rotation around the Z-axis.
        Rz: np.ndarray = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz

    def intrinsic(self, info: CameraInfo) -> None:
        """
        Extract and compute the camera intrinsic matrix and its inverse.

        Parameters:
            info (CameraInfo): Message containing the intrinsic parameters of the camera.
        """
        # Convert the intrinsic parameters list into a 3x3 NumPy array.
        self.K = np.array(info.k).reshape(3, 3)
        # Compute the inverse of the intrinsic matrix.
        self.K_inv = np.linalg.inv(self.K)
        self.get_logger().info(f'Received camera intrinsics:\n{self.K}')

    def rayplane_intersection(self) -> Optional[Tuple[float, float, float]]:
        """
        Compute the intersection point of the ray (through the pixel) with the ground plane.

        The method uses the projection matrix to compute a direction vector (dw) for the ray.
        The intersection is computed under the assumption that the ground plane is at Z = 0.

        Returns:
            Optional[Tuple[float, float, float]]: (X, Y, Z) coordinates of the intersection point,
            or None if the computation fails.
        """
        if self.P is None or self.pixels is None or self.altitude is None:
            self.get_logger().warn("Required parameters for ray-plane intersection are not set.")
            return None
        
        # Compute the direction vector in the camera frame by projecting the pixel.
        dw: np.ndarray = self.P @ self.pixels

        # Avoid division by zero by ensuring the Z component is significant.
        if np.abs(dw[2]) < 1e-6:
            self.get_logger().warn("dw[2] is too close to zero, cannot compute intersection.")
            return None

        # Compute the scaling factor 't' to reach the ground plane (Z = 0).
        self.t = self.altitude / dw[2]
        
        # Calculate the intersection point.
        X: float = self.t * dw[0] + self.x
        Y: float = self.t * dw[1] + self.y
        Z: float = 0.0  # Ground plane

        self.get_logger().info(f'Calculated world coordinates \n X: {X}, Y: {Y}, Z: {Z} ')
        
        return (X, Y, Z)

    def publish_waypoint(self) -> None:
        """
        Publish the computed world coordinates as a PoseStamped message.

        The position is set using the intersection point, and the orientation is
        set to the identity quaternion (indicating no rotation).
        """
        coords: Optional[Tuple[float, float, float]] = self.rayplane_intersection()
        if coords is None:
            self.get_logger().warn("No valid intersection computed, not publishing waypoint.")
            return

        waypoint: PoseStamped = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.pose.position.x = coords[0]
        waypoint.pose.position.y = coords[1]
        waypoint.pose.position.z = coords[2]

        # Set the orientation to the identity quaternion (no rotation).
        waypoint.pose.orientation.x = 0.0
        waypoint.pose.orientation.y = 0.0
        waypoint.pose.orientation.z = 0.0
        waypoint.pose.orientation.w = 1.0

        # Publish the computed waypoint.
        self.waypoint_pub.publish(waypoint)
        self.get_logger().info("Published new waypoint.")

def main() -> None:
    """
    Main entry point for the transform node.

    Initializes the ROS node and keeps it spinning until a shutdown signal is received.
    """
    rclpy.init()
    transform_node = TransformNode()

    try:
        rclpy.spin(transform_node)
    except KeyboardInterrupt:
        transform_node.get_logger().info("Shutting down transform_node")
    finally:
        transform_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




