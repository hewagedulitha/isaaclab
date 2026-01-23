import gymnasium as gym
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg,ViewerCfg
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

filter_prim_path_expr = [
    ]

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
class CubeTerrainLoopEnvSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(
                            prim_path="{ENV_REGEX_NS}/outback", 
                            spawn=sim_utils.UsdFileCfg(
                                usd_path="/home/hewaged/Documents/cube_terrain_loop_nav.usd",
                            )
            )

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
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
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
                        pos=(-115.0, -132.0, 0.4), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
            # pos=(-24.0, 24.0, 0.4), rot=(0.70711, 0.0, 0.0, -0.70711), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
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

    #Tiled Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1/chassis_link/front_cam",
        # update_period=0.1,
        height=100,
        width=100,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:mud": ( 110, 69, 47, 255),
            "class:asphalt": (55, 8, 209, 255),
            "class:obstacle": (19, 42, 69, 255),
            "class:dirt": (26,177, 219, 255),
            "class:grass": (18, 224, 87, 255),
                                       },
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.86), rot=(0.57206, 0.41563, -0.41563, -0.57206), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.86), rot=(0.57206, 0.41563, -0.41563, -0.57206), convention="opengl"),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),

    )

   

@configclass
class CubeTerrainLoopEnvCfg(DirectRLEnvCfg):
    # scene
    scene: CubeTerrainLoopEnvSceneCfg = CubeTerrainLoopEnvSceneCfg(num_envs=1, env_spacing=192.0, replicate_physics=False)

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0), lookat=(-108.0, -132.0, 0.0))

    # events
    # events: EventCfg = EventCfg()

    # robot
    # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    # )

    # env
    episode_length_s = 400.0

    #DQN
    # decimation = 8

    #SAC
    decimation = 4

    action_scale = 2.0
    #use normalized action spaces for PPO. Not required if using SAC in which case, action_space = 1 is used
    # action_space = gym.spaces.Box(low=float(-0.5), high=float(0.5), shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=float(-1.0), high=float(1.0), shape=(1,), dtype=np.float32)
    
    # MultiDict action spaces are not supported by SAC (https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
    # action_space = gym.spaces.Dict({
    #     "linear": gym.spaces.Box(low=float(0.0), high=float(0.5), shape=(1,), dtype=np.float32),
    #     "angular": gym.spaces.Box(low=float(-1.0), high=float(1.0), shape=(1,), dtype=np.float32),
    # })
    
    #dqn discrete action space
    # action_space = gym.spaces.Discrete(3)

    # spaces
    observation_space = gym.spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(scene.camera.height, scene.camera.width, 3)
    )  # or for simplicity: [height, width, 3]

    # observation_space = gym.spaces.Box(
    #         low=-np.inf,
    #         high=np.inf,
    #         shape=(32 + 20,), #encoded RGB data + lidar points
    #         dtype=np.float32,
    #     )  # or for simplicity: [height, width, 3]
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, #DQN 1/200, SAC 1/100
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    #terrain cube coords
    OFFSET = 152.0
    yellow_cube_pos_x = [
            [-147.5, -144.5], # left, 1
            [-147.5, -144.5], # left, 2
            [-147.5, -144.5], # left, 3
            [-147.5 + OFFSET, -144.5 + OFFSET], # right, 1
            [-147.5 + OFFSET, -144.5 + OFFSET], # right, 2
            [-147.5 + OFFSET, -144.5 + OFFSET], # right, 3
            [-142.0, -106.0], # bottom, 1
            [-82.0, -58.5], # bottom, 2
            [-34.0, -2.0], # bottom, 3
            [-142.0, -106.0], # top, 1
            [-82.0, -58.5], # top, 2
            [-34.0, -2.0], # top, 3
        ]
    yellow_cube_pos_y = [
            [-128.0, -92.0], #left 1
            [-67.5, -44.0], #left 2
            [-20.0, 16.0], #left 3
            [-128.0, -92.0], #right 1
            [-67.5, -44.0], #right 2
            [-20.0, 16.0], #right 3
            [-133.5, -130.5], #bottom 1
            [-133.5, -130.5], #bottom 2
            [-133.5, -130.5], #bottom 3
            [-133.5 + OFFSET, -130.5 + OFFSET], #top 1
            [-133.5 + OFFSET, -130.5 + OFFSET], #top 2
            [-133.5 + OFFSET, -130.5 + OFFSET], #top 3
        ]
    blue_cube_pos_x = [
            [-155.5, -152.5], #left long 1
            [-151.5, -146.5], #left short 1
            [-151.5, -146.5], #left short 2
            [-139.5, -136.5], #left long 2
            [-145.5, -140.5], #left short 3
            [-145.5, -140.5], #left short 4
            [-155.5 + OFFSET, -152.5 + OFFSET], #right long 1
            [-152.5 + OFFSET, -146.5 + OFFSET], #right short 1
            [-152.5 + OFFSET, -146.5 + OFFSET], #right short 2
            [-139.5 + OFFSET, -136.5 + OFFSET], #right long 2
            [-145.5 + OFFSET, -140.5 + OFFSET], #right short 3
            [-145.5 + OFFSET, -140.5 + OFFSET], #right short 4
            [-105.5, -82.5], #bottom long 1
            [-105.5, -102.5], #bottom short 1
            [-85.5, -82.5], #bottom short 2
            [-57.5, -34.5], #bottom long 2
            [-57.5, -54.5], #bottom short 3
            [-37.5, -34.5], #bottom short 4
            [-105.5, -82.5], #top long 1
            [-105.5, -102.5], #top short 1
            [-85.5, -82.5], #top short 2
            [-57.5, -34.5], #top long 2
            [-57.5, -54.5], #top short 3
            [-37.5, -34.5], #top short 4
        ]
    blue_cube_pos_y = [
            [-91.5, -68.5], #left long 1
            [-91.5, -88.5], #left short 1
            [-71.5, -68.5], #left short 2
            [-43.5, -20.5], #left long 2
            [-43.5, -40.5], #left short 3
            [-23.5, -20.5], #left short 4
            [-91.5, -68.5], #right long 1
            [-91.5, -88.5], #right short 1
            [-71.5, -68.5], #right short 2
            [-43.5, -20.5], #right long 2
            [-43.5, -40.5], #right short 3
            [-23.5, -20.5], #right short 4
            [-125.5, -122.5], #bottom long 1
            [-131.5, -126.5], #bottom short 1
            [-131.5, -126.5], #bottom short 2
            [-141.5, -138.5], #bottom long 2
            [-137.5, -132.5], #bottom short 3
            [-137.5, -132.5], #bottom short 4
            [-125.5 + OFFSET, -122.5 + OFFSET], #top long 1
            [-131.5 + OFFSET, -126.5 + OFFSET], #top short 1
            [-131.5 + OFFSET, -126.5 + OFFSET], #top short 2
            [-141.5 + OFFSET, -138.5 + OFFSET], #top long 2
            [-137.5 + OFFSET, -132.5 + OFFSET], #top short 3
            [-137.5 + OFFSET, -132.5 + OFFSET], #top short 4
        ]
    red_cube_pos_x = [
            [-139.5, -136.5], #left long 1
            [-145.5, -140.5], #left short 1
            [-145.5, -140.5], #left short 2
            [-155.5, -152.5], #left long 2
            [-151.5, -146.5], #left short 3
            [-151.5, -146.5], #left short 4
            [-139.5 + OFFSET, -136.5 + OFFSET], #right long 1
            [-145.5 + OFFSET, -140.5 + OFFSET], #right short 1
            [-145.5 + OFFSET, -140.5 + OFFSET], #right short 2
            [-155.5 + OFFSET, -152.5 + OFFSET], #right long 2
            [-151.5 + OFFSET, -146.5 + OFFSET], #right short 3
            [-151.5 + OFFSET, -146.5 + OFFSET], #right short 4
            [-105.5, -82.5], #bottom long 1
            [-105.5, -102.5], #bottom short 1
            [-85.5, -82.5], #bottom short 2
            [-57.5, -34.5], #bottom long 2
            [-57.5, -54.5], #bottom short 3
            [-37.5, -34.5], #bottom short 4
            [-105.5, -82.5], #top long 1
            [-105.5, -102.5], #top short 1
            [-85.5, -82.5], #top short 2
            [-57.5, -34.5], #top long 2
            [-57.5, -54.5], #top short 3
            [-37.5, -34.5], #top short 4
        ]
    red_cube_pos_y = [
            [-91.5, -68.5], #left long 1
            [-91.5, -88.5], #left short 1
            [-71.5, -68.5], #left short 2
            [-43.5, -20.5], #left long 2
            [-23.5, -20.5], #left short 3
            [-43.5, -40.5], #left short 4
            [-91.5, -68.5], #right long 1
            [-91.5, -88.5], #right short 1
            [-71.5, -68.5], #right short 2
            [-43.5, -20.5], #right long 2
            [-23.5, -20.5], #right short 3
            [-43.5, -40.5], #right short 4
            [-141.5, -138.5], #bottom long 1
            [-137.5, -132.5], #bottom short 1
            [-137.5, -132.5], #bottom short 2
            [-125.5, -122.5], #bottom long 2
            [-131.5, -126.5], #bottom short 3
            [-131.5, -126.5], #bottom short 4
            [-141.5 + OFFSET, -138.5 + OFFSET], #top long 1
            [-137.5 + OFFSET, -132.5 + OFFSET], #top short 1
            [-137.5 + OFFSET, -132.5 + OFFSET], #top short 2
            [-125.5 + OFFSET, -122.5 + OFFSET], #top long 2
            [-131.5 + OFFSET, -126.5 + OFFSET], #top short 3
            [-131.5 + OFFSET, -126.5 + OFFSET], #top short 4
        ]

    # reward scales
    goal_reward_scale = 2000.0
    clash_reward_scale = -1000.0
    goal_distance_reward_scale = 1.0
    terrain_rewards = [0.3, 0.2, 0.1, -100.0] # yellow, blue, red, green
