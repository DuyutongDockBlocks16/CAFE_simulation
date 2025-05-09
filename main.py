import gymnasium as gym
import my_env  # 确保你的环境类已实现并在同目录
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
import mujoco.viewer
import time

# 注册环境
gym.register(
    id="MyMuJoCoEnv-v0",
    entry_point="my_env:MyMuJoCoEnv",
    kwargs={"xml_path": "scene.xml"}
)

# 定义渲染回调
class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=10):
        super().__init__()
        self.env = env
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        return True

def model_training(env):
    # model = PPO.load("ppo_mujoco_car", env=env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300_000, callback=RenderCallback(env))
    model.save("ppo_mujoco_car")
    env.close()

def model_implementation(env):
    model = PPO.load("ppo_mujoco_car", env=env)
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

    model = env.unwrapped.model
    data = env.unwrapped.data

    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
    
    env.close()

    # 设置仿真步长
    time_step = 0.001
    model.opt.timestep = time_step  # 将自定义步长应用到模型

    # 启动被动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC to exit viewer...")
        last_time = time.time()
        frame_count = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                print(f"Simulated FPS: {frame_count}")
                frame_count = 0
                last_time = now


if __name__ == "__main__":
    env = gym.make("MyMuJoCoEnv-v0")
    # model_training(env)
    model_implementation(env)