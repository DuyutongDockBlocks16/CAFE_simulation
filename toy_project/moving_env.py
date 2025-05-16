import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class MovingMuJoCoEnv(gym.Env):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        print("Model ID in MovingMuJoCoEnv:", id(model))
        print("Data ID in MovingMuJoCoEnv:", id(data))
        actuator_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        print("Data of actuator:", data.ctrl[actuator_index])
        # 定义 observation_space 和 action_space
        mujoco.mj_forward(self.model, self.data)
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_area")


        self.prev_dist = None
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        num_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(num_actuators - 1,),  # 减少一个维度
            dtype=np.float32
        )
        self.max_steps = 10000
        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        target_obj_pos = self.data.xpos[self.target_body_id]
        target_area_pos = self.model.geom_pos[self.target_area_geom_id]
        # print("Object0 position (xpos):", self.data.xpos[self.target_body_id])
        # print("Target area position (xpos):", self.data.xpos[self.target_area_id])
        
        self.prev_dist = np.linalg.norm(target_obj_pos - target_area_pos)
        return self._get_obs(), {}


    def step(self, action):
        full_action = np.zeros(self.model.nu)  # 创建一个与 self.data.ctrl 形状匹配的数组
        full_action[:len(action)] = action  # 将 action 的值填入前部分
        actuator_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        full_action[actuator_index] = self.data.ctrl[actuator_index]  # 保留原始值
        self.data.ctrl[:] = full_action  # 赋值给 self.data.ctrl
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        target_obj_pos = self.data.xpos[self.target_body_id]
        target_area_pos = self.model.geom_pos[self.target_area_geom_id]
        dist = np.linalg.norm(target_obj_pos - target_area_pos)
        reward = (self.prev_dist - dist) * 100
        if reward < 0:
            reward -= 0.1 * self.current_step / 1000  # 每步都减去0.01 * 当前步数，表示时间惩罚
        else:
            reward += 0.01 * self.current_step / 1000

        

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        if truncated:
            print(f"truncated")
            reward -= 500
        else:
            reward -= (dist - self.prev_dist) * self.current_step * 10  # 距离惩罚

        self.prev_dist = dist
        terminated = dist < 0.1
        if terminated:
            print(f"terminated")
            reward += 10000000

        info = {}
        print(f"Distance: {dist}, Reward: {reward}")
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 返回观测向量
        target_obj_pos = self.data.xpos[self.target_body_id]
        target_area_pos = self.model.geom_pos[self.target_area_geom_id]
        return np.concatenate([target_obj_pos, target_area_pos]).astype(np.float32)

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()