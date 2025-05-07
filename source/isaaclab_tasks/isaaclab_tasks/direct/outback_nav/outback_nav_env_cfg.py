# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg

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


##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    box_pos = EventTerm(
        func=mdp.randomize_cube_pos,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_collection"),
        },
    )

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
            scale=(2.0, 2.0, 2.0),
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

    # Camera
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/front_cam",
    #     update_period=0.1,
    #     height=100,
    #     width=100,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    #Tiled Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/front_cam",
        # update_period=0.1,
        height=80,
        width=80,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    #LIDAR - name is important. Otherwise, the sensor will get initialized before the robot, resulting in an error.
    camera_lidar = LidarCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/camera_mount/lidar",
        # update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        horizontal_fov=270.0,  # Horizontal field of view of 270 degrees
        horizontal_resolution=0.2497,  # Horizontal resolution of 0.5 degrees
        max_range=30.0,  # Maximum range of 30 meters
        min_range=0.02,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        vertical_fov=60.0,
        vertical_resolution=4.0,
        spawn=sim_utils.LidarSpawnCfg(),
        offset=LidarCfg.OffsetCfg(
            pos=(0.11749, 0.0, 4.0),  # Example position offset from the robot base
            
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=False,
        draw_points=False,
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

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

@configclass
class OutbackNavEnvCfg(DirectRLEnvCfg):
    # scene
    scene: MultiObjectSceneCfg = MultiObjectSceneCfg(num_envs=1, env_spacing=72.0, replicate_physics=False)

    # events
    events: EventCfg = EventCfg()

    # robot
    # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    # )

    # env
    episode_length_s = 20.0
    decimation = 100
    action_scale = 2.0
    #use normalized action spaces for PPO. Not required if using SAC in which case, action_space = 1 is used
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(scene.camera.height, scene.camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # reward scales
    goal_reward_scale = 20.0
    clash_reward_scale = -10.0
    goal_distance_reward_scale = 10.0
