# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
import wandb
import cv2
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCollection,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from ae.autoencoder import load_ae


from .outback_maze_env_cfg import OutbackMazeNavEnvCfg


class OutbackMazeNavEnv(DirectRLEnv):
    cfg: OutbackMazeNavEnvCfg

    def __init__(self, cfg: OutbackMazeNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_reward",
                "clash_reward",
                "goal_distance_reward",
            ]
        }

        wheel_radius = 0.48
        wheel_base = 1.08
        self._controller: DifferentialController = DifferentialController("test_controller", wheel_radius, wheel_base, max_angular_speed=1.0)

        # self._run = wandb.init(
        #     # Set the wandb entity where your project will be logged (generally your team name).
        #     entity="hewaged-edith-cowan-university",
        #     # Set the wandb project where this run will be logged.
        #     project="outback-nav-ppo",
        #     # Track hyperparameters and run metadata.
        #     config={
        #         "max_episode_length (seconds)": self.max_episode_length_s,
        #         "algo": "PPO",
        #         "sim_dt": self.cfg.sim.dt,
        #         "decimation": self.cfg.decimation,
        #         "num_envs": self.num_envs,
        #     },
        # )
        self.autoencoder = load_ae(os.environ["AE_PATH"])


    def _setup_scene(self):
        self._robot = self.scene["robot"]
        # self.scene.articulations["robot"] = self._robot
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor
        # if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            # self._height_scanner = RayCaster(self.cfg.height_scanner)
            # self.scene.sensors["height_scanner"] = self._height_scanner
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        # self.scene.clone_environments(copy_from_source=False)
        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        # print(f"[INFO]:self._actions Shape: {self._actions.shape} self.cfg.action_scale: {self.cfg.action_scale}")
        self._processed_actions = self._actions
        linear_speed = 2.0
        # print(f"[INFO]:_processed_actions: {self._processed_actions} shape: {self._processed_actions.shape}")
        out = torch.cat([linear_speed * torch.ones((self.num_envs, 1), device=self.device), self._processed_actions], 1)
        joint_vel_targets = torch.stack([self.get_joint_vel_targets(i) for i in out ])
        joint_vel = self._robot.data.default_joint_vel.clone()
        joint_vel[...,:2] = joint_vel_targets
        # print(f"Controller max_angular_speed:{self._controller.max_angular_speed} actions: {out} joint_vel_targets: {joint_vel_targets} #####################")
        self._robot.set_joint_velocity_target(joint_vel)
    
    def get_joint_vel_targets(self, command: torch.Tensor) -> torch.Tensor:
        command_np = command.clone().detach().cpu().numpy()
        articulation_actions = self._controller.forward(command_np)
        return torch.tensor(articulation_actions.joint_velocities)


    def _apply_action(self):
        linear_speed = 1.0
        # print(f"[INFO]:_processed_actions Shape: {self._processed_actions.shape} self.num_envs: {self.num_envs}")
        # out = torch.cat([linear_speed * torch.ones((self.num_envs, 1), device=self.device), self._processed_actions], 1)
        # joint_vel_targets = torch.stack([self.get_joint_vel_targets(i) for i in out ])
        # joint_vel = self._robot.data.default_joint_vel.clone()
        # joint_vel[...,:2] = joint_vel_targets
        # print(f"Controller max_angular_speed:{self._controller.max_angular_speed} actions: {out} joint_vel_targets: {joint_vel_targets} #####################")
        # self._robot.set_joint_velocity_target(joint_vel)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        lidar = self.scene["camera_lidar"]
        # print(f"[INFO]: RTX Lidar Output Shape: {lidar.data.output.shape} RTX Lidar Output:{lidar.data.output}")
        camera = self.scene["camera"]
        # print("[INFO]: Camera Output Shape: ", camera.data.output["rgb"].shape)
        # print(scene["camera"])
        # print("Received shape of semantic_segmentation   image: ", camera.data.output["semantic_segmentation"].shape)
        # print("Received shape of distance_to_image_plane image: ", camera.data.output["distance_to_image_plane"].shape)

        sem_seg = torch.squeeze(camera.data.output["semantic_segmentation"][..., :3].clone(), dim=0).cpu().detach().numpy()
        # print("Processed shape of sem_seg: ", sem_seg.shape)
        # print("Processed sem_seg: ", sem_seg)
        # cwd = os.getcwd()
        # print("Current directory: ",cwd)
        # cv2.imshow("sem_seg_1.png", sem_seg)
        # cv2.waitKey(1)
        # save all camera RGB images
        # cam_images = scene["camera"].data.output["rgb"][..., :3]
        encoded_image = self.autoencoder.encode_from_raw_image(sem_seg)
        obs = torch.unsqueeze(torch.tensor(encoded_image.flatten(), device=self.device, dtype=torch.float), 0)

        # reconstructed_image = self.autoencoder.decode(encoded_image)[0]
        # cv2.imshow("Original", sem_seg)
        # cv2.imshow("Reconstruction", reconstructed_image)
        # cv2.waitKey(1)
        
        # img = cam_images[0].detach().cpu().numpy()

        # camera_data = self.scene["camera"].data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        # mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        # camera_data -= mean_tensor    
        # print("[INFO]: camera_data Shape: ", camera_data.shape)                                      
        # obs = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_observation_space["policy"]), device=self.device)
        # obs = torch.flatten(camera.data.output["semantic_segmentation"].clone(), start_dim=1,).float()
        # print("Processed obs shape: ", obs.shape)
        # print("Processed obs: ", obs)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        body_pose = self.scene["robot"].data.body_pos_w.clone()[:, 0]
        # print(f"[INFO]: body_pose Shape: {body_pose.shape} RTX Lidar Output:{body_pose}")

        goals = self.scene.env_origins.clone() + torch.tensor([24.0, -24.0, 0.0], dtype=torch.float, device=self.device)

        # distance error
        distance_to_the_goal = torch.sqrt((body_pose[:, 0]-goals[:, 0])**2 + (body_pose[:, 1]-goals[:, 1])**2)
        distance_to_the_goal_error = torch.maximum(1 - distance_to_the_goal/67.22, torch.zeros(self.num_envs, dtype=torch.float, device=self.device))

        #clash error
        force_matrices_L = self.scene["contact_sensor_L"].data.force_matrix_w.clone()
        force_matrices_R = self.scene["contact_sensor_R"].data.force_matrix_w.clone()
        force_matrices = torch.cat((force_matrices_L, force_matrices_R), dim=-1)
        flat_force_matrices = torch.flatten(force_matrices, start_dim=1)
        # max_force = torch.max(flat_force_matrices, dim=1, keepdim=True)[0]
        died = torch.any(flat_force_matrices != 0.0, dim=1)
        clash_error=torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        clash_error[died] = 1.0
        # if died:
        #     #in_contact
        #     clash_error = torch.tensor(1.0, dtype=torch.float, device=self.device)
        
        #goal
        x_in_goal = torch.logical_and(20.0 < body_pose[:, 0], body_pose[:, 0]  < 28.0) 
        y_in_goal =  torch.logical_and(-20.0 > body_pose[:, 1], body_pose[:, 1]  > -28.0)
        in_goal = torch.logical_and(x_in_goal, y_in_goal)
        goal_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        goal_error[in_goal] = 1.0

        rewards = {
            "goal_reward": goal_error * self.cfg.goal_reward_scale,
            "clash_reward": clash_error * self.cfg.clash_reward_scale,
            "goal_distance_reward": distance_to_the_goal_error * self.cfg.goal_distance_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # print(f"[INFO]: clash_error: {clash_error}")
        # print(f"[INFO]: goal_distance_error: {distance_to_the_goal_error} ")
        # print(f"[INFO]: goal_reward: {goal_error} ")
        # print(f"rewards: {rewards} reward sum:{reward}")
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        body_pose = self.scene["robot"].data.body_pos_w.clone()[:, 0]
        # print(f"[INFO]: body_pose : {body_pose}")
        x_in_goal = torch.logical_and(20.0 < body_pose[:, 0], body_pose[:, 0]  < 28.0) 
        y_in_goal =  torch.logical_and(-20.0 > body_pose[:, 1], body_pose[:, 1]  > -28.0)
        in_goal = torch.logical_and(x_in_goal, y_in_goal)
        # print(f"[INFO]: in_goal : {in_goal}")

        time_out = torch.logical_or(time_out, in_goal)
        force_matrices_L = self.scene["contact_sensor_L"].data.force_matrix_w.clone()
        force_matrices_R = self.scene["contact_sensor_R"].data.force_matrix_w.clone()
        force_matrices = torch.cat((force_matrices_L, force_matrices_R), dim=-1)
        # net_forces = torch.cat((self.scene["contact_sensor_L"].data.net_forces_w, self.scene["contact_sensor_R"].data.net_forces_w), dim=-1)
        # net_forces_w_history = torch.cat((self.scene["contact_sensor_L"].data.net_forces_w_history, self.scene["contact_sensor_R"].data.net_forces_w_history), dim=-1)
        # last_contact_time = torch.cat((self.scene["contact_sensor_L"].data.last_contact_time, self.scene["contact_sensor_R"].data.last_contact_time), dim=-1)
        # current_contact_time = torch.cat((self.scene["contact_sensor_L"].data.current_contact_time, self.scene["contact_sensor_R"].data.current_contact_time), dim=-1)
        flat_force_matrices = torch.flatten(force_matrices, start_dim=1)
        # print(f"[INFO]: flat_force_matrices: {flat_force_matrices}")
        # print(f"[INFO]: net_forces : {net_forces}")
        # print(f"[INFO]: net_forces_w_history : {net_forces_w_history}")
        # print("[INFO]: last_contact_time: ", self.scene["contact_sensor_L"].data.last_contact_time)
        # print(f"[INFO]: current_contact_time : {current_contact_time}")
        # max_force = torch.max(flat_force_matrices, dim=1, keepdim=True)[0]
        # print(f"[INFO]: force_matrices_L: {force_matrices_L.shape} force_matrices_R:{force_matrices_R.shape} force_matrices:{force_matrices.shape} flat_force_matrices:{flat_force_matrices.shape} max_force: {max_force.shape}")
        died = torch.any(flat_force_matrices != 0.0, dim=1)
        if died or time_out:
            print(f"[INFO]: self.episode_length_buf : {self.episode_length_buf}  self.max_episode_length :{ self.max_episode_length }")
            print(f"[INFO]: _get_dones died: {died} time_out:{time_out}")
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # print("[INFO] Resetting ")
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            # print("[INFO] Resetting all indices")
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Reset robot state
        root_state = self._robot.data.default_root_state.clone()
        # print(f"[INFO]: root_state: {root_state}")
        root_state[env_ids, :3] += self.scene.env_origins[env_ids, :]
        joint_pos, joint_vel = self._robot.data.default_joint_pos.clone(), self._robot.data.default_joint_vel.clone()
        self._robot.write_root_pose_to_sim(root_state[env_ids, :7], env_ids= env_ids)
        self._robot.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids= env_ids)
        self._robot.write_joint_state_to_sim(joint_pos[env_ids, :], joint_vel[env_ids, :], env_ids= env_ids)
        

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        self._run.log(self.extras["log"])
    
    def close(self):
        self._run.finish()
        super().close()
        
