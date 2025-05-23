# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
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
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


@configclass
class CubeMazeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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
            pos=(0.0, 24.0, 2.0), rot=(0.70711, 0.0, 0.0, -0.70711), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0}
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

    static_cube1 = AssetBaseCfg(
                    prim_path="/World/envs/env_.*/S_Cube_1", 
                    spawn=sim_utils.CuboidCfg(
                        size=(8.0, 8.0*3, 8.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
                        # ),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", "cube")],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(16.0, 24.0, 4.0)),
            )
    static_cube2 = AssetBaseCfg(
                    prim_path="/World/envs/env_.*/S_Cube_2", 
                    spawn=sim_utils.CuboidCfg(
                        size=(8.0*3, 8.0, 8.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
                        # ),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", "cube")],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -16.0, 4.0)),
            )
    static_cube3 = AssetBaseCfg(
                    prim_path="/World/envs/env_.*/S_Cube_3", 
                    spawn=sim_utils.CuboidCfg(
                        size=(8.0, 8.0*6, 8.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
                        # ),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", "cube")],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(40.0, -12.0, 4.0)),
            )
    static_cube4 = AssetBaseCfg(
                    prim_path="/World/envs/env_.*/S_Cube_4", 
                    spawn=sim_utils.CuboidCfg(
                        size=(8.0, 8.0*2, 8.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
                        # ),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", "cube")],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -28.0, 4.0)),
            )
    static_cube5 = AssetBaseCfg(
                    prim_path="/World/envs/env_.*/S_Cube_5", 
                    spawn=sim_utils.CuboidCfg(
                        size=(8.0*2, 8.0, 8.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
                        # ),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        semantic_tags=[("class", "cube")],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(16.0, 8.0, 4.0)),
            )
    # cube_2 = AssetBaseCfg(
    #             prim_path="/World/envs/env_.*/Cube_2",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0*2, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                 # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                 # ),
    #                 # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, 20.0, 4.0)),
    #         )
    # cube_3 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_3",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0*4, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(20.0, 8.0, 4.0)),
    #             )
    # cube_4 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_4",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*5, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(32.0, -16.0, 4.0)),
    #             )
    # cube_5 = AssetBaseCfg(             prim_path="/World/envs/env_.*/Cube_5",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(24.0, -32.0, 4.0)),
    #             )
    # cube_6 = AssetBaseCfg(             prim_path="/World/envs/env_.*/Cube_6",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*4, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(16.0, -20.0, 4.0)),
    #             )
    # cube_7 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_7",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0*3, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -8.0, 4.0)),
    #             )
    # cube_8 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_8",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*2, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(-8.0, 4.0, 4.0)),
    #             )
    # cube_9 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_9",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0*4, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(-20.0, 16.0, 4.0)),
    #             )
    # cube_10 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/Cube_10",
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                         # solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True,
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(-32.0, 24.0, 4.0)),
    #             )

    # object collection
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
    
            "cube_1": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_1",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0*6, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-12.0, 32.0, 4.0)),
            ),
                "cube_2": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_2",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0*2, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 20.0, 4.0)),
            ),
               "cube_3": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_3",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0*3, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, 8.0, 4.0)),
            ),
               "cube_4": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_4",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0*5, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(32.0, -16.0, 4.0)),
            ),
            #    "cube_5": RigidObjectCfg(
            #     prim_path="/World/envs/env_.*/Cube_5",
            #     spawn=sim_utils.CuboidCfg(
            #         size=(8.0, 8.0, 8.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #             solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
            #         ),
            #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #         collision_props=sim_utils.CollisionPropertiesCfg(),
            #         semantic_tags=[("class", "cube")],
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, -32.0, 4.0)),
            # ),
               "cube_6": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_6",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0*3, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(16.0, -24.0, 4.0)),
            ),
               "cube_7": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_7",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0*3, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -8.0, 4.0)),
            ),
               "cube_8": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_8",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0*2, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, 4.0, 4.0)),
            ),
               "cube_9": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_9",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0*4, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-28.0, 16.0, 4.0)),
            ),
               "cube_10": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube_10",
                spawn=sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 8.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    semantic_tags=[("class", "cube")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-32.0, 24.0, 4.0)),
            ),
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
        height=480,
        width=640,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={"class:cube": (25, 255, 140, 255),},
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
            ],
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # robot = scene["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # -- write data to sim
        # scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        # scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CubeMazeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
