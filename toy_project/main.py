import gymnasium as gym
import approach_env 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
import mujoco.viewer
import time

APPROACHING_MODEL_NAME = "ppo_mujoco_car_20250509_ready"
MOVING_MODEL_NAME = "ppo_moving_car"

# 注册环境
gym.register(
    id="ApproachMuJoCoEnv-v0",
    entry_point="approach_env:ApproachMuJoCoEnv",
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

def approach_model_training(env):
    model = PPO.load("ppo_mujoco_car", env=env)
    # model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300_000, callback=RenderCallback(env))
    model.save("ppo_mujoco_car")
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

    model = env.unwrapped.model
    data = env.unwrapped.data

    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
    
    env.close()

    # 设置仿真步长
    time_step = 0.001
    model.opt.timestep = time_step  # 将自定义步长应用到模型

    actuator_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    data.ctrl[actuator_index] = 255

    return data, model



def moving_model_training(env):
    model = PPO("MlpPolicy", env, verbose=1)
    # model = PPO.load("ppo_moving_car", env=env)

    start_time = time.time()  # 记录训练开始时间
    max_training_time = 3 * 60 * 60  # 三小时 = 10800秒

    while True:
        model.learn(total_timesteps=300_000, callback=RenderCallback(env))  # 每次训练一小段时间
        elapsed_time = time.time() - start_time  # 计算已用时间
        if elapsed_time >= max_training_time:
            print("Training completed: Reached 3 hours.")
            break

    model.save("ppo_moving_car")
    env.close()

def moving_model_implementation(env):

    model = PPO.load(MOVING_MODEL_NAME, env=env)
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

    actuator_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    data.ctrl[actuator_index] = 0

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
                # print(f"Simulated FPS: {frame_count}")
                frame_count = 0
                last_time = now


if __name__ == "__main__":
    approach_env = gym.make("ApproachMuJoCoEnv-v0")
    # approach_model_training(approach_env)
    data, model = approach_model_implementation(approach_env)

    # actuator_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    # print("Data of actuator:", data.ctrl[actuator_index])

    target_area_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_area")
    print("Target area geom_pos:", model.geom_pos[target_area_geom_id])

    print("Model ID:", id(model))
    print("Data ID:", id(data))

    gym.register(
        id="MovingMuJoCoEnv-v0",
        entry_point="moving_env:MovingMuJoCoEnv",
        kwargs={"model": model, "data": data}
    )

    moving_env = gym.make("MovingMuJoCoEnv-v0")
    # moving_model_training(moving_env)
    moving_model_implementation(moving_env)

    