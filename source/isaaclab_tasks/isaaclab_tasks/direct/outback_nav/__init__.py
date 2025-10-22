# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Nav-Maze-Outback-Carter-V1-Direct-v0",
    entry_point=f"{__name__}.cube_maze_nav_env:CubeMazeNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_maze_env_cfg:CubeMazeNavEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_dqn_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0",
    entry_point=f"{__name__}.outback_nav_env:OutbackNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.outback_nav_env_cfg:OutbackNavEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_dqn_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cube-Maze-Terrain-Nav-Direct-v0",
    entry_point=f"{__name__}.cube_maze_terrain_env:CubeMazeTerrainNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_maze_terrain_env_cfg:CubeMazeTerrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sim2Real-Cube-Nav-Direct-v0",
    entry_point=f"{__name__}.sim2real_cube_env:Sim2RealCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_cube_env_cfg:Sim2RealCubeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)
