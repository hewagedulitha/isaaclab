#!/bin/bash

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0 --num_envs 1 --enable_cameras --seed 84 --replay_buffer replay_buffer_1170

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0 --num_envs 1 --enable_cameras --seed 63 --replay_buffer replay_buffer_1170

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0 --num_envs 1 --enable_cameras --seed 42 --replay_buffer replay_buffer_1170

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0 --num_envs 1 --enable_cameras --seed 21 --replay_buffer replay_buffer_1170

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Terrain-Nav-Outback-Carter-V1-Direct-v0 --num_envs 1 --enable_cameras --seed 11 --replay_buffer replay_buffer_1170
