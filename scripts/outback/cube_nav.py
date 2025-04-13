'from __future__ import annotations

# -------------- Launch Isaac Sim Simulator first. -------------------


import argparse
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument('--env-name', default="obstacle_avoidance",
                    help='quadruped_isaac')
parser.add_argument('--policy', default="Gaussian",
                help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                help='Temperature parameter α determines the relative importance of the entropy\
                        term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=4000, metavar='N',
                help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=500000, metavar='N',
                help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                help='run on CUDA (default: False)')

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -------------- Rest of the application. -------------------

import cv2
import rclpy
from rclpy.node import Node
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import datetime
import copy
import os
import math
from math import sin, cos, atan2

import carb
import omni.usd
from pxr import Gf, Sdf, PhysxSchema, UsdPhysics
import isaaclab.sim as sim_utils
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaacsim.core.api import World
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg, LidarCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image

from sac import SAC
from replay_memory import ReplayMemory

##

# Scene Configuration

##

@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/carter_v1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-24.0, 24.0, 2.0), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0}
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness={"left_wheel": 0.0, "right_wheel": 0.0},
                damping={"left_wheel": 10000.0, "right_wheel": 10000.0},
            ),
        },
    )

    # Camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    #LIDAR
    lidar: LidarCfg = LidarCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/camera_mount",
        # update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        horizontal_fov=270.0,  # Horizontal field of view of 270 degrees
        horizontal_resolution=0.2497,  # Horizontal resolution of 0.5 degrees
        max_range=30.0,  # Maximum range of 30 meters
        min_range=0.02,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        vertical_fov=60.0,
        vertical_resolution=4.0,
        offset=LidarCfg.OffsetCfg(
            pos=(0.11749, 0.0, 4.0),  # Example position offset from the robot base
            
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=True,
        draw_points=True,
    )
    
    # contact sensors
    contact_sensor_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/left_wheel_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
            "{ENV_REGEX_NS}/Cube_3",
            "{ENV_REGEX_NS}/Cube_4",
            "{ENV_REGEX_NS}/Cube_5",
            "{ENV_REGEX_NS}/Cube_6",
            "{ENV_REGEX_NS}/Cube_7",
            "{ENV_REGEX_NS}/Cube_8",
            "{ENV_REGEX_NS}/Cube_9",
            "{ENV_REGEX_NS}/Cube_10",
            "{ENV_REGEX_NS}/Cube_11",
            "{ENV_REGEX_NS}/Cube_12",
            "{ENV_REGEX_NS}/Cube_13",
            "{ENV_REGEX_NS}/Cube_14",
            ],
    )

    contact_sensor_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/right_wheel_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
            "{ENV_REGEX_NS}/Cube_3",
            "{ENV_REGEX_NS}/Cube_4",
            "{ENV_REGEX_NS}/Cube_5",
            "{ENV_REGEX_NS}/Cube_6",
            "{ENV_REGEX_NS}/Cube_7",
            "{ENV_REGEX_NS}/Cube_8",
            "{ENV_REGEX_NS}/Cube_9",
            "{ENV_REGEX_NS}/Cube_10",
            "{ENV_REGEX_NS}/Cube_11",
            "{ENV_REGEX_NS}/Cube_12",
            "{ENV_REGEX_NS}/Cube_13",
            "{ENV_REGEX_NS}/Cube_14",
            ],
    )

    #LIDAR
    # /World/carter_v1/chassis_link/Rotating

    
    # ray_caster = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
    #     mesh_prim_paths=["/World/defaultGroundPlane"],
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
    #     ),
    #     debug_vis=True,
    # )

    # rigid object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             sim_utils.ConeCfg(
    #                 radius=0.3,
    #                 height=0.6,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(0.3, 0.3, 0.3),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.SphereCfg(
    #                 radius=0.3,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
    #             ),
    #         ],
    #         random_choice=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    # )


    # object collection
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            # "object_A": RigidObjectCfg(
            #     prim_path="/World/envs/env_.*/Object_A",
            #     spawn=sim_utils.SphereCfg(
            #         radius=0.1,
            #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
            #         ),
            #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #         collision_props=sim_utils.CollisionPropertiesCfg(),
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
            # ),
            "cube_1": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_1",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, 32.0, 4.0)),
            ),
            "cube_2": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_2",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 24.0, 4.0)),
            ),
               "cube_3": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_3",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(16.0, 32.0, 4.0)),
            ),
               "cube_4": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_4",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, 24.0, 4.0)),
            ),
               "cube_5": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_5",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-32.0, 8.0, 4.0)),
            ),
               "cube_6": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_6",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-24.0, 0.0, 4.0)),
            ),
               "cube_7": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_7",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, 8.0, 4.0)),
            ),
               "cube_8": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_8",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
            ),
               "cube_9": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_9",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(32.0, 8.0, 4.0)),
            ),
               "cube_10": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_10",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, 0.0, 4.0)),
            ),
               "cube_11": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_11",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-32.0, -16.0, 4.0)),
            ),
               "cube_12": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_12",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-24.0, -24.0, 4.0)),
            ),
               "cube_13": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_13",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, -16.0, 4.0)),
            ),
               "cube_14": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_14",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -24.0, 4.0)),
            ),
            
            # "object_C": RigidObjectCfg(
            #     prim_path="/World/envs/env_.*/Object_C",
            #     spawn=sim_utils.ConeCfg(
            #         radius=0.1,
            #         height=0.3,
            #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
            #         ),
            #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #         collision_props=sim_utils.CollisionPropertiesCfg(),
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 2.0)),
            # ),
        }
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    # axes = axes.flatten()

    img = images[0].detach().cpu().numpy()
    ax.imshow(img, cmap=cmap)
    ax.axis("off")

    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()

def randomized_box_pos():
    boxes_pos = [
                    [[-8, 32, 4], [0.0, 32, 4], [8, 32, 4], [-8, 24, 4], [0.0, 24, 4], [8, 24, 4], [-8, 16, 4], [0.0, 16, 4], [8, 16, 4],],
                    [[16, 32, 4], [24, 32, 4], [32, 32, 4], [16, 24, 4], [24, 24, 4], [32, 24, 4], [16, 16, 4], [24, 16, 4], [32, 16, 4],],
                    [[-32, 8, 4], [-24, 8, 4], [-16, 8, 4], [-32, 0.0, 4], [-24, 0.0, 4], [-16, 0.0, 4], [-32, -8, 4], [-24, -8, 4], [-16, -8, 4]],
                    [[-8, 8, 4], [0.0, 8, 4], [8, 8, 4], [-8, 0.0, 4], [0.0, 0.0, 4], [8, 0.0, 4], [-8, -8, 4], [0.0, -8, 4], [8, -8, 4],],
                    [[32, 8, 4], [24, 8, 4], [16, 8, 4], [32, 0.0, 4], [24, 0.0, 4], [16, 0.0, 4], [32, -8, 4], [24, -8, 4], [16, -8, 4],],
                    [[-32, -16, 4], [-24, -16, 4], [-16, -16, 4], [-32, -24, 4], [-24, -24, 4], [-16, -24, 4], [-32, -32, 4], [-24, -32, 4], [-16, -32, 4],],
                    [[-8, -16, 4], [0.0, -16, 4], [8, -16, 4], [-8, -24, 4], [0.0, -24, 4], [8, -24, 4], [-8, -32, 4], [0.0, -32, 4], [8, -32, 4],],
                ]

    box_pos_tensor = []
    for i in range(14):
        index = np.random.randint(9)
        box_pos_tensor.append(boxes_pos[int(i/2)][index])

    return torch.tensor(box_pos_tensor).unsqueeze(0)

def step(scene: InteractiveScene, action, lidar_data, prev_dist, time_steps, max_episode_steps):

    body_pose = scene["robot"].data.body_pos_w.clone()[0, 0].cpu().detach().numpy()
    force_matrices_L = scene["contact_sensor_L"].data.force_matrix_w.clone()[0, 0].detach().cpu().numpy()
    force_matrices_R = scene["contact_sensor_R"].data.force_matrix_w.clone()[0, 0].detach().cpu().numpy()

    in_collision=False
    for force_matrix in force_matrices_L:
        if force_matrix[0] != 0.0 or force_matrix[0] != 0.0 or force_matrix[0] != 0.0:
            in_collision = True
    
    for force_matrix in force_matrices_R:
        if force_matrix[0] != 0.0 or force_matrix[0] != 0.0 or force_matrix[0] != 0.0:
            in_collision = True

    goal = [24.0, -24.0]
    next_state = np.zeros(22) # 0~19:lidar, 20,21:x,y distance to the goal
    done = False
    omega = 2*action[0]

    distance_to_the_goal = math.sqrt((body_pose[0]-goal[0])**2 + (body_pose[1]-goal[1])**2)

    next_state[:20] = lidar_data/30
    next_state[20] = -(body_pose[0]- goal[0])/48 
    next_state[21] = (body_pose[1]- goal[1])/48

    if(in_collision):
        print(f"[INFO] CLASH. DONE.")
        done = True
        reward = -10
    # elif(time_steps >= max_episode_steps):
    #    print(f"[INFO] TIME OUT. DONE.")
    #    done = True    
    #    reward = 0        
    elif( 20.0 < body_pose[0] < 28.0 and 
            -28.0 < body_pose[1] < -20.0):
        print(f"[INFO] GOAL. DONE.")
        done = True           
        reward = 20
    elif(min(lidar_data) < 3):
        print(f"[INFO] Too close to the wall.")
        reward = -5
    elif(distance_to_the_goal < prev_dist):
        reward = 10*max(1 - distance_to_the_goal/67.22, 0)
    else:
        reward = -1

    prev_dist = distance_to_the_goal

    print(f"[INFO] self.next_state:{next_state}, collision:{in_collision}, reward:{reward}, omega:{omega}")
    # self.get_logger().info(f"Robot Proximity data: {self.proximity_sensor.get_data()}")
    # for sensor in self.proximity_sensors:
    #     data = sensor.get_data()
    #     self.get_logger().info(f"Proximity data: {data}")
    #     if "/World/omni_robot/base_footprint/base_link" in data:
    #         self.get_logger().info(f"Proximity data: {data}")
    #     # carb.log_warn(f"prox data: {data}")

    return prev_dist, next_state, reward, done


##
# Simulation Loop
##

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # Agent
    action_space = np.zeros(1) # velocity, and steering_angle
    agent = SAC(len(np.zeros(22)), action_space, args_cli)

    # Memory
    memory = ReplayMemory(args_cli.replay_size, args_cli.seed)

    # Extract scene entities
    # note: we only do this here for readability.
    # rigid_object: RigidObject = scene["object"]
    rigid_object_collection: RigidObjectCollection = scene["object_collection"]
    robot: Articulation = scene["robot"]

    # Training Loop
    total_numsteps = 0
    updates = 0
    max_episode_steps = 120
    prev_dist = 0
    in_collision = False
    lidar_data = np.zeros(20)
    episode_reward = 0
    episode_steps = 0
    done = False

    # Controller
    wheel_radius = 0.24
    wheel_base = 0.54
    controller = DifferentialController("test_controller", wheel_radius, wheel_base)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    #Sensors
    triggered = False
    countdown = 42

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Simulation loop
    while simulation_app.is_running():

        

        state = np.zeros(22)

        if count % 20 == 0:
            print(f"episode step:{episode_steps}, in total:{total_numsteps}/{args_cli.num_steps}")

            if args_cli.start_steps > total_numsteps:
                action = 2*np.random.rand(len(action_space)) - 1 # -1 <= action <= 1
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args_cli.batch_size:            
                for i in range(args_cli.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args_cli.batch_size, updates)
                    updates += 1

            # Apply action to robot
            linear_speed = 1.0
            angular_speed = 2*action[0]
            command = [linear_speed, angular_speed]
            articulation_actions = controller.forward(command)
            joint_vel = robot.data.default_joint_vel.clone()
            joint_vel[...,:2] = torch.tensor(articulation_actions.joint_velocities)
            lidar = scene["lidar"]
            print(f"[INFO]: RTX Lidar Output Shape: {lidar.data.output.shape} RTX Lidar Output:{lidar.data.output}")
            print(f"ROBOT JOINT NAMES:{robot.data.joint_names} JOINT VEL: {joint_vel} #####################")
            robot.set_joint_velocity_target(joint_vel)


            # print information from the sensors
            # print("-------------------------------")
            # print(scene["contact_sensor"])
            # print("Received force matrix of: ", scene["contact_sensor"].data.force_matrix_w)
            # print("Received contact force of: ", scene["contact_sensor"].data.net_forces_w)

            # if not triggered:
            #     if countdown > 0:
            #         countdown -= 1
            #         continue
            #     data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
            #     np.save("cast_data.npy", data)
            #     triggered = True

            # print("-------------------------------")
            # print(scene["camera"])
            # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
            # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)

            # save all camera RGB images
            # cam_images = scene["camera"].data.output["rgb"][..., :3]
            
            # img = cam_images[0].detach().cpu().numpy()
            # filename = os.path.join(output_dir, "cam_rgb", f"{count:04d}.jpg")
            # print(f"Received shape of rgb   image: {img.shape}, filename: {filename}")
            # cv2.imwrite(filename, img)
            # save_images_grid(
            #     cam_images,
            #     subtitles=[f"Cam{i}" for i in range(cam_images.shape[0])],
            #     title="Camera RGB Image",
            #     filename=filename,
            # )
            
            prev_dist, next_state, reward, done = step(scene, action, lidar_data, prev_dist, episode_steps, max_episode_steps) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
        
        # Reset
        if done:

            # reset counter
            # count = 0
            episode_reward = 0
            episode_steps = 0
            done = False

            # reset the scene entities
            # object
            # root_state = rigid_object.data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            # rigid_object.write_root_pose_to_sim(root_state[:, :7])
            # rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

            # object collection
            object_state = rigid_object_collection.data.default_object_state.clone()
            # print(f"[INFO]: Object state: {object_state[..., :3]}")
            # print(f"[INFO]: Scene origins: {scene.env_origins.unsqueeze(1).shape}")
            # object_state[..., :3] += scene.env_origins.unsqueeze(1)
            object_state[..., :3] = randomized_box_pos()
            rigid_object_collection.write_object_link_pose_to_sim(object_state[..., :7])
            rigid_object_collection.write_object_com_velocity_to_sim(object_state[..., 7:])

            # robot
            # -- root state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # -- joint state
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()

            print("[INFO]: Resetting scene state...")

        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Increment counter
        count += 1

        # Update buffers
        scene.update(sim_dt)

        if total_numsteps > args_cli.num_steps:
            print(f"[INFO]: Total Number of Steps Achieved")
            break



def main():
    """Main function."""

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 50)

    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # with Timer("[INFO] Time to randomize scene: "):
    #     # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
    #     # Note: Just need to acquire the right attribute about the property you want to set
    #     # Here is an example on setting color randomly
    #     randomize_shape_color(scene_cfg.object.prim_path)

    # stage = omni.usd.get_context().get_stage()
    # # Add a physics scene prim to stage
    # phys_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    # # Set gravity vector
    # phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    # phys_scene.CreateGravityMagnitudeAttr().Set(981.0)

    # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
    # physxSceneAPI.CreateEnableCCDAttr(True)
    # physxSceneAPI.CreateEnableStabilizationAttr(True)
    # physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    # physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    # physxSceneAPI.CreateSolverTypeAttr("TGS")

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)



if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
