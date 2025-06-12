import gymnasium as gym
import sec_robot_env 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
import mujoco.viewer
import time


gym.register(
    id="SecondRobotMuJoCoEnv-v0",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={"xml_path": "scene_mirobot.xml"}
)

APPROACHING_MODEL_NAME = "ppo_mujoco_car_10M.zip"

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
    # Create PPO model with training parameters
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=3e-4,     # Learning rate
                n_steps=2048,           # Collect 2048 steps of experience each time
                batch_size=64,          # Process 64 samples per batch
                tensorboard_log="./ppo_logs/")  # Log save path
    
    # Train for 10 iterations, 1 million steps each
    for i in range(10):  # Total 10 million steps
        model.learn(total_timesteps=1_000_000,      # Train 1 million steps each time
                # callback=RenderCallback(env),    # Render callback
                reset_num_timesteps=False)       # Don't reset timestep counter
        model.save(f"ppo_mujoco_car_{i+1}M")       # Save model
        print(f"Saved model at {(i+1)}M steps")
    
    env.close()

def approach_model_implementation(env):
    model = PPO.load(APPROACHING_MODEL_NAME, env=env)
    obs, info = env.reset()
    for _ in range(200000000000):
        env.render()  # 每步都渲染
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # obs, info = env.reset()
            env.unwrapped.data.ctrl[:] = 0
            mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  # 执行一步仿真以应用归零信号
            break
    
    # env.close()


if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    approach_model_training(approach_env)
    # approach_model_implementation(approach_env)
