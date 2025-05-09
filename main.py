import gymnasium as gym
import my_env  # 确保你的环境类已实现并在同目录
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

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

if __name__ == "__main__":
    env = gym.make("MyMuJoCoEnv-v0")

    # model = PPO.load("ppo_mujoco_car", env=env)
    # # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=300_000, callback=RenderCallback(env))
    # model.save("ppo_mujoco_car")
    # env.close()

    model = PPO.load("ppo_mujoco_car", env=env)
    obs, info = env.reset()
    for _ in range(200000000000):
        env.render()  # 每步都渲染
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()