from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.utils.math import convert_camera_frame_orientation_convention

@dataclass
class LidarData:
    """Data container for the LiDAR sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # LiDAR data
    ##


    # output: TensorDict = None
    output: torch.Tensor  = None
    """The retrieved sensor data with sensor types as key.

    For LiDAR, this might include processed data such as point clouds or enhanced distance measurements,
    depending on the sensor's capabilities and the processing applied.
    """

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the sensor aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following
        OpenGL / USD Camera convention.

        .. note::
            OpenGL convention follows the sensor aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="opengl")
