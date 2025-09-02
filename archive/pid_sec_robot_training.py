import gymnasium as gym
import sec_robot_env 
from stable_baselines3 import PPO
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

gym.register(
    id="SecondRobotMuJoCoEnv-v1",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={"xml_path": "xml/scene_mirobot.xml"}
)

if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v1")
    approach_model_training(approach_env)