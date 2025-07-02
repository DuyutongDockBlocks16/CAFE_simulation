import gymnasium as gym
import sac_second_robot_env 
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from time import sleep
import mujoco.viewer
import time
import os
from datetime import datetime
from config.env_config import Direction, Layer, FiniteState
from config.training_config import APPROACHING_MODEL_NAME, SUCCESS_THRESHOLD
from callbacks.episode_data_collector import EpisodeBatchCollector
from callbacks.success_check_point_saver import SuccessCheckpointCallback
from callbacks.training_renderer import RenderCallback
import numpy as np

gym.register(
    id="SacSecondRobotMuJoCoEnv-v0",
    entry_point="sac_second_robot_env:SacSecondRobotMuJoCoEnv",
    kwargs={"xml_path": "xml/scene_mirobot.xml"}
)

n_sampled_goal = 4

def approach_model_training(env):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    combined_callback = CallbackList([
        RenderCallback(env),
        SuccessCheckpointCallback("./checkpoints"),
        episode_collector
    ])

    model = SAC(
            "MultiInputPolicy", 
            env, 
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            learning_starts=6000
        ) 
    model.learn(
        total_timesteps=int(2e6),
        callback=combined_callback,
    )
        

    model.save("her_sac_highway")


if __name__ == "__main__":
    approach_env = gym.make("SacSecondRobotMuJoCoEnv-v0")
    approach_model_training(approach_env)