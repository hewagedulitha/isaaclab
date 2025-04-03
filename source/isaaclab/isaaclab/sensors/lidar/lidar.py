# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence

from typing import TYPE_CHECKING, Literal

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.usd
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

# from omni.isaac.core.utils.extensions import enable_extension

# enable_extension("omni.isaac.range_sensor")  # required by OIGE
# enable_extension("omni.isaac.RangeSensorSchema")  # required by OIGE
from isaacsim.sensors.physx import _range_sensor
import omni.isaac.RangeSensorSchema as RangeSensorSchema

import isaaclab.sim as sim_utils
from isaaclab.utils import to_camel_case
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

from ..sensor_base import SensorBase
from .lidar_data import LidarData

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg


class Lidar(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A rendered color image.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"instance_segmentation"``: The instance segmentation data.
    - ``"semantic_segmentation"``: The semantic segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.

        In case you need to work with these sensor types, we recommend using the single camera implementation
        from the :mod:`omni.isaac.orbit.compat.camera` module.

    .. _replicator extension: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    cfg: LidarCfg
    """The configuration parameters."""
    # UNSUPPORTED_TYPES: set[str] = {"bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d"}
    """The set of sensor types that are not supported by the camera class."""

    def __init__(self, cfg: LidarCfg):
        
        """Initializes the camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the camera sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # perform check on supported data types
        # self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        
        # Create empty variables for storing output data
        self._data = LidarData()
        self._li = _range_sensor.acquire_lidar_sensor_interface()
        
        self.num_beams = int(cfg.horizontal_fov / cfg.horizontal_resolution)
        self.num_lines = int(cfg.vertical_fov / cfg.vertical_resolution) + 1

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe callbacks
        super().__del__()
        # delete from replicator registry

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Lidar @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            # f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> LidarData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame


    """
    Configuration
    """

    def set_lidar_properties(self):
        """"If any value is set to None, the default value is used."""

        env_ids = self._ALL_INDICES

        # Iterate over environment IDs
        for i in env_ids:
            # Get corresponding lidar prim path
            lidar = RangeSensorSchema.Lidar.Define(omni.usd.get_context().get_stage(), self._view.prim_paths[i])

            if self.cfg.horizontal_fov is not None:
                lidar.GetHorizontalFovAttr().Set(self.cfg.horizontal_fov)
            
            if self.cfg.horizontal_resolution is not None:
                lidar.GetHorizontalResolutionAttr().Set(self.cfg.horizontal_resolution)
                
            if self.cfg.max_range is not None:
                lidar.GetMaxRangeAttr().Set(self.cfg.max_range)
            
            if self.cfg.min_range is not None:
                lidar.GetMinRangeAttr().Set(self.cfg.min_range)
                
            if self.cfg.rotation_rate is not None:
                lidar.GetRotationRateAttr().Set(self.cfg.rotation_rate)
            
            if self.cfg.vertical_fov is not None:
                lidar.GetVerticalFovAttr().Set(self.cfg.vertical_fov)
            
            if self.cfg.vertical_resolution is not None:
                lidar.GetVerticalResolutionAttr().Set(self.cfg.vertical_resolution)
                
            if self.cfg.draw_lines is not None:
                lidar.GetDrawLinesAttr().Set(self.cfg.draw_lines)
                
            if self.cfg.draw_points is not None:
                lidar.GetDrawPointsAttr().Set(self.cfg.draw_points)
                

            # Additional LiDAR-specific properties can be set here as needed.

    """
    Operations - Set pose.
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        r"""Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`omni.isaac.orbit.sensors.camera.utils.convert_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # convert to backend tensor
        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)
        # convert rotation matrix from input convention to OpenGL
        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")
        # set the pose
        self._view.set_world_poses(positions, orientations, env_ids)

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is (N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # set camera poses using the view
        orientations = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, device=self._device))
        self._view.set_world_poses(eyes, orientations, env_ids)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs randomization on the camera poses.
        self._update_poses(env_ids)
        # self._update_intrinsic_matrices(env_ids)
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the LiDAR sensor handles and internal buffers.

        This function prepares the LiDAR sensor for data collection, ensuring it is properly configured within the simulation environment. It also initializes the internal buffers to store the LiDAR data.

        Raises:
            RuntimeError: If the number of LiDAR prims in the view does not match the expected number.
        """
        import omni.replicator.core as rep

        # Initialize the base class
        super()._initialize_impl()

        # Prepare a view for the LiDAR sensor based on its path
        pos = torch.tensor(self.cfg.offset.pos, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
                rot, origin=self.cfg.offset.convention, target="opengl"
            )
        self._view = XFormPrim(self.cfg.prim_path, translations=pos, reset_xform_properties=False)
        self._view.initialize()

        # Ensure the number of detected LiDAR prims matches the expected number
        if self._view.count != self._num_envs:
            raise RuntimeError(f"Expected number of LiDAR prims ({self._num_envs}) does not match the found number ({self._view.count}).")

        # Prepare environment ID buffers
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)

        # Initialize a frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        # Resolve device name
        if "cuda" in self._device:
            device_name = self._device.split(":")[0]
        else:
            device_name = "cpu"
        
        self.set_lidar_properties()
 
        # Create internal buffers for LiDAR data
        self._create_buffers()

    # TODO : Make this check
    def _is_valid_lidar_prim(self, prim):
        # Checking if a USD prim is a valid LiDAR sensor in simulation environment.
        return True


    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1
        # -- pose
        self._update_poses(env_ids)

        for index in env_ids:
            linear_depth = self._li.get_linear_depth_data(self._view.prim_paths[index])
            output = convert_to_torch(linear_depth, device=self.device)
            self._data.output[index] = output.squeeze()

    """
    Private Helpers
    """

    def _create_buffers(self):
        """Create buffers for storing LiDAR distance measurement data."""
        # Pose of the LiDAR sensors in the world
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._data.output = torch.zeros((self._view.count, self.num_beams, self.num_lines), device=self._device)
        

    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """

        # get the poses from the view
        poses, quat = self._view.get_world_poses(env_ids)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(quat, origin="opengl", target="world")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
