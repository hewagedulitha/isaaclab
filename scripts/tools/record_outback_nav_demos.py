# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch
import wandb
import json
import numpy as np
import pathlib
import pickle
from typing import Any, Optional, Union
import io

import omni.log
from stable_baselines3.common.buffers import ReplayBuffer

from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.envs import (
    DirectRLEnv,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)

def process_extras(obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray, num_envs: int) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(num_envs)]

    for idx in range(num_envs):
        infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
        if idx in reset_ids:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
        else:
            infos[idx]["terminal_observation"] = None
    # return list of dictionaries
    return infos

def open_path(
    path: Union[str, pathlib.Path, io.BufferedIOBase], mode: str, verbose: int = 0, suffix: Optional[str] = None
) -> Union[io.BufferedWriter, io.BufferedReader, io.BytesIO, io.BufferedRandom]:
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.

    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose >= 2,
    it raises a warning.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    # Note(antonin): the true annotation should be IO[bytes]
    # but there is not easy way to check that
    allowed_types = (io.BufferedWriter, io.BufferedReader, io.BytesIO, io.BufferedRandom)
    if not isinstance(path, allowed_types):
        raise TypeError(f"Path {path} parameter has invalid type: expected one of {allowed_types}.")
    if path.closed:
        raise ValueError(f"File stream {path} is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError as e:
        raise ValueError("Expected mode to be either 'w' or 'r'.") from e
    if (("w" == mode) and not path.writable()) or (("r" == mode) and not path.readable()):
        error_msg = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {error_msg} file.")
    return path

def save_to_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], obj: Any, verbose: int = 0) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    with open(str(path), 'wb') as fp:
        # file = open_path(path, "w", verbose=verbose, suffix="pkl")
    # Use protocol>=4 to support saving replay buffers >= 4Gb
    # See https://docs.python.org/3/library/pickle.html
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # if isinstance(path, (str, pathlib.Path)):
    #     file.close()

def save_replay_buffer(replay_buffer: ReplayBuffer, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
    """
    Save the replay buffer as a pickle file.

    :param path: Path to the file where the replay buffer should be saved.
        if path is a str or pathlib.Path, the path is automatically created if necessary.
    """
    
    save_to_pkl(path, replay_buffer, 1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.teleop_device.lower() == "handtracking":
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    # if hasattr(env_cfg.terminations, "success"):
    #     success_term = env_cfg.terminations.success
    #     env_cfg.terminations.success = None
    # else:
    #     omni.log.warn(
    #         "No success termination term was found in the environment."
    #         " Will not be able to mark recorded demos as successful."
    #     )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    # env_cfg.terminations.time_out = None

    # env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if isinstance(env, DirectRLEnv):
        direct_env: DirectRLEnv = env
        wandb_run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="hewaged-edith-cowan-university",
                # Set the wandb project where this run will be logged.
                project="outback-terrain-demo",
                # Track hyperparameters and run metadata.
                config={
                    "rl_library": "sb3",
                },
                sync_tensorboard=True,
            )
        direct_env._run = wandb_run

    # add teleoperation key for reset current recording instance
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=1.0, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", reset_recording_instance)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking'."
        )

    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset before starting
    obs, _ = env.reset()
    obs = obs["policy"].detach().cpu().numpy()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    demo_count = 0
    sim_step_counter = 0
    transition_counter = 0
    state_dicts = []

    replay_buffer = ReplayBuffer(buffer_size=200000, 
                                 observation_space=env.observation_space,
                                 action_space=env.action_space,
                                 device=env.device,
                                 n_envs=env.num_envs,
                                 )

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = delta_pose[:, 1]

            if np.any(actions.detach().cpu().numpy() == 3.0):
                print(f"[INFO] delta_pose: {delta_pose} actions: {actions}")
                continue
            # print(f"[INFO] actions: {actions}")
            # perform action on environment
            prev_obs = obs
            obs, rew, terminated, truncated, extras = env.step(actions)
            target = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
            target = torch.where(actions == 1.0, -0.5, target)
            target = torch.where(actions == 2.0, 0.5, target)
            actions = target.detach().cpu().numpy()
            # print(f"[INFO] actions: {actions} shape: {actions.shape}")
            dones = terminated | truncated
            reset_ids = (dones > 0).nonzero(as_tuple=False)

            if np.any(actions == 3.0):
                print(f"[INFO] delta_pose: {delta_pose} actions: {actions}")

            obs = obs["policy"].detach().cpu().numpy()
            rew = rew.detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
            dones = dones.detach().cpu().numpy()

            infos = process_extras(obs, terminated, truncated, extras, reset_ids, env.num_envs)

            state_dicts.append({
                "obs" : prev_obs,
                "next_obs": obs,
                "action": actions,
                "reward": rew,
                "done": dones,
                "infos": infos,
            })

            replay_buffer.add(
                prev_obs,
                obs,
                actions,
                rew,
                dones,
                infos,
            )

            transition_counter += 1

            if dones:
                print(f"[INFO] saving json with {transition_counter} transitions in buffer")
                # save_replay_buffer(replay_buffer, f'datasets/replay_buffer.pkl')
                # with open(f'datasets/result_{transition_counter}.json', 'w') as fp:
                #     json.dump(state_dicts, fp)
                #     demo_count += 1


            if should_reset_recording_instance:
                print(f"[INFO] saving replay buffer with {transition_counter} transitions")

                duplicate_count = transition_counter

                while duplicate_count < 50000:
                    for d in state_dicts:
                        replay_buffer.add(d["obs"], d["next_obs"], d["action"], d["reward"], d["done"], d["infos"])
                        duplicate_count += 1

                save_replay_buffer(replay_buffer, f'datasets/cube_maze_terrain_sac_replay_buffer_{transition_counter}.pkl')
                # with open(f'datasets/result_{transition_counter}.json', 'w') as fp:
                #     json.dump(state_dicts, fp)
                should_reset_recording_instance = False
                success_step_count = 0

            # print out the current demo count if it has changed
            # if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
            #     current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            #     print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            # if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
            #     print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
            #     break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

             #perform physics stepping
            for _ in range(env.cfg.decimation):
                sim_step_counter += 1
                env.sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if sim_step_counter % env.cfg.sim.render_interval == 0:
                    env.sim.render()
                # update buffers at sim dt
                env.scene.update(dt=env.physics_dt)

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
