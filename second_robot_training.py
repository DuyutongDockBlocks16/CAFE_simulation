import gymnasium as gym
import sec_robot_env 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
import mujoco.viewer
import time


gym.register(
    id="SecondRobotMuJoCoEnv-v0",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={"xml_path": "scene_mirobot.xml"}
)

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=10):
        super().__init__()
        self.env = env
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        return True

def approach_model_training(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300_000, callback=RenderCallback(env))
    model.save("ppo_mujoco_car")
    env.close()

if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    approach_model_training(approach_env)
