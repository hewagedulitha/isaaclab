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
        "{ENV_REGEX_NS}/outback/SM_Tree_L_01/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_02/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_01/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_02/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_Log_M_01_01/SM_Log_M_01",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_03/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_03/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_04/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_05/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_06/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_04/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_Log_M_01_02/SM_Log_M_01",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_05/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_07/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_08/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_06/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_Log_M_01_03/SM_Log_M_01",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_07/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_09/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_10/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_11/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_08/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_Log_M_01_05/SM_Log_M_01",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_09/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_Log_M_01_04/SM_Log_M_01",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_10/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_12/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_13/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_14/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05_15/SM_GrassTree_04",
            "{ENV_REGEX_NS}/outback/SM_Tree_L_11/SM_Tree_L_01/Section0",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_02/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_Log_L_01/SM_Log_L_01",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_04/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_03/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_05/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_06/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_07/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_Log_L_03/SM_Log_L_01",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_08/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_09/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_10/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_Log_L_02/SM_Log_L_01",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_11/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_12/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_Log_L_04/SM_Log_L_01",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_13/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_14/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_15/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_16/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_17/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_18/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_19/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_20/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_21/SM_GrassTree_02/Section1",
            "{ENV_REGEX_NS}/outback/SM_GrassTree_22/SM_GrassTree_02/Section1",
    ] + ["{ENV_REGEX_NS}/outback/" + f"SM_GrassTree_05_{x}/SM_GrassTree_04" for x in range(16, 53) ] + ["{ENV_REGEX_NS}/outback/" + f"SM_GrassTree_0{x}/SM_GrassTree_02/Section0" for x in range(2, 10) ] + ["{ENV_REGEX_NS}/outback/" + f"SM_GrassTree_{x}/SM_GrassTree_02/Section0" for x in range(10, 23) ]


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
class OutbackNavEnvSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/outback", 
                          spawn=sim_utils.UsdFileCfg(
                            usd_path="/home/hewaged/Documents/sac_nav_carter.usd",
                            ))

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1_2",
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
                        pos=(-24.0, 24.0, 0.6), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
            # pos=(24.0, -18.0, 2.0), rot=(0.70711, 0.0, 0.0, -0.70711), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
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

    # static_cube1 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_1", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*3, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(16.0, 24.0, 4.0)),
    #         )
    # static_cube2 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_2", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0*3, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -16.0, 4.0)),
    #         )
    # static_cube3 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_3", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*6, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(40.0, -12.0, 4.0)),
    #         )
    # static_cube4 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_4", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*2, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -28.0, 4.0)),
    #         )
    # static_cube5 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_5", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0*2, 8.0, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(12.0, -40.0, 4.0)),
    #         )
    # static_cube6 = AssetBaseCfg(
    #                 prim_path="/World/envs/env_.*/S_Cube_6", 
    #                 spawn=sim_utils.CuboidCfg(
    #                     size=(8.0, 8.0*2, 8.0),
    #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     #     solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #                     # ),
    #                     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                     collision_props=sim_utils.CollisionPropertiesCfg(),
    #                     semantic_tags=[("class", "cube")],
    #                 ),
    #                 init_state=AssetBaseCfg.InitialStateCfg(pos=(-16.0, 4.0, 4.0)),
    #         )
    
    # object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    #     rigid_objects={
    
    #         "cube_1": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_1",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0*5, 0.5, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(-16.0, 28.0, 4.0)),
    #         ),
    #         "cube_2": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_2",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(0.5, 8.0*3, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 16.0, 4.0)), # close turn 2 pos=(8.0, 16.0, 4.0)
    #         ),
    #         #  "cube_corner_2": RigidObjectCfg(
    #         #     prim_path="/World/envs/env_.*/Cube_Corner_2",
    #         #     spawn=sim_utils.CuboidCfg(
    #         #         size=(8.0, 8.0, 8.0),
    #         #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #         #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #             solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #         #         ),
    #         #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         #         semantic_tags=[("class", "cube")],
    #         #     ),
    #         #     init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 8.0, 4.0)),
    #         # ),
    #            "cube_3": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_3",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0*3, 8.0, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, 8.0, 4.0)),
    #         ),
    #            "cube_4": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_4",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0*5, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(32.0, -16.0, 4.0)),
    #         ),
    #         #    "cube_5": RigidObjectCfg(
    #         #     prim_path="/World/envs/env_.*/Cube_5",
    #         #     spawn=sim_utils.CuboidCfg(
    #         #         size=(8.0, 8.0, 8.0),
    #         #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #         #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #             solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #         #         ),
    #         #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         #         semantic_tags=[("class", "cube")],
    #         #     ),
    #         #     init_state=RigidObjectCfg.InitialStateCfg(pos=(24.0, -32.0, 4.0)),
    #         # ),
    #         "cube_6": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_6",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0*3, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(16.0, -24.0, 4.0)),
    #         ),
    #         # "cube_corner_3": RigidObjectCfg(
    #         #     prim_path="/World/envs/env_.*/Cube_Corner_3",
    #         #     spawn=sim_utils.CuboidCfg(
    #         #         size=(8.0, 8.0, 8.0),
    #         #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #         #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #             solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #         #         ),
    #         #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         #         semantic_tags=[("class", "cube")],
    #         #     ),
    #         #     init_state=RigidObjectCfg.InitialStateCfg(pos=(16.0, -8.0, 4.0)),
    #         # ),
    #         "cube_7": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_7",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0*4, 8.0, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.0, -8.0, 4.0)), # close pos=(4.0, -8.0, 4.0)
    #         ),
    #         "cube_8": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_8",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0*2, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, 4.0, 4.0)),
    #         ),
    #         #  "cube_corner_1": RigidObjectCfg(
    #         #     prim_path="/World/envs/env_.*/Cube_Corner_1",
    #         #     spawn=sim_utils.CuboidCfg(
    #         #         size=(8.0, 8.0, 8.0),
    #         #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #         #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #             solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #         #         ),
    #         #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         #         semantic_tags=[("class", "cube")],
    #         #     ),
    #         #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.0, 16.0, 4.0)),
    #         # ),
    #            "cube_9": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_9",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0*4, 8.0, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(-28.0, 16.0, 4.0)), # close pos=(-20.0, 16.0, 4.0)
    #         ),
    #            "cube_10": RigidObjectCfg(
    #             prim_path="/World/envs/env_.*/Cube_10",
    #             spawn=sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0, 8.0),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=1.0),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=False,
    #                 ),
    #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #                 collision_props=sim_utils.CollisionPropertiesCfg(),
    #                 semantic_tags=[("class", "cube")],
    #             ),
    #             init_state=RigidObjectCfg.InitialStateCfg(pos=(-32.0, 24.0, 4.0)),
    #         ),
    #     }
    # )


    #Tiled Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1_2/chassis_link/front_cam",
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
        prim_path="{ENV_REGEX_NS}/carter_v1_2/chassis_link/camera_mount/lidar",
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
        prim_path="{ENV_REGEX_NS}/carter_v1_2/left_wheel_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=filter_prim_path_expr,            

    )

    contact_sensor_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1_2/right_wheel_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=filter_prim_path_expr,
    )

    contact_sensor_C = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/carter_v1_2/chassis_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=filter_prim_path_expr,
    )

   

@configclass
class OutbackNavEnvCfg(DirectRLEnvCfg):
    # scene
    scene: OutbackNavEnvSceneCfg = OutbackNavEnvSceneCfg(num_envs=1, env_spacing=72.0, replicate_physics=False)

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0), lookat=(-24.0, 24.0, 0.0))

    # events
    # events: EventCfg = EventCfg()

    # robot
    # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    # )

    # env
    episode_length_s = 80.0

    #DQN
    decimation = 8

    #SAC
    # decimation = 4

    action_scale = 2.0
    #use normalized action spaces for PPO. Not required if using SAC in which case, action_space = 1 is used
    # action_space = gym.spaces.Box(low=float(-0.5), high=float(0.5), shape=(1,), dtype=np.float32)
    
    #dqn discrete action space
    action_space = gym.spaces.Discrete(3)

    observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32,
        )  # or for simplicity: [height, width, 3]
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, #DQN 1/200, SAC 1/100
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
    goal_distance_reward_scale = 1.0
