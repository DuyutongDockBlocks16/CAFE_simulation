import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class MyMuJoCoEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # 定义 observation_space 和 action_space
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.right_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
        self.left_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")
        self.prev_dist = None
        self.target_init_pos = None
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.model.nu,), dtype=np.float32
        )
        self.max_steps = 10000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        right_pad_pos = self.data.xpos[self.right_pad_id]
        left_pad_pos = self.data.xpos[self.left_pad_id]
        # position of middle of the two pads
        middle = (right_pad_pos + left_pad_pos) / 2
        target_pos = self.data.xpos[self.target_body_id]
        self.prev_dist = np.linalg.norm(middle - target_pos)  # 初始化距离
        self.target_init_pos = np.copy(([-8.30454536e-19 , 1.00000000e+00 , 3.97844892e-02]))
        return self._get_obs(), {}


    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        # 获取小车和目标的位置
        right_pad_pos = self.data.xpos[self.right_pad_id]
        left_pad_pos = self.data.xpos[self.left_pad_id]
        middle = (right_pad_pos + left_pad_pos) / 2
        target_pos = self.data.xpos[self.target_body_id]
        dist = np.linalg.norm(middle - target_pos)
        reward = (self.prev_dist - dist) * 100
        if reward < 0:
            reward -= 0.01 * self.current_step / 1000  # 每步都减去0.01 * 当前步数，表示时间惩罚
        else:
            reward += 0.01 * self.current_step / 1000

        target_moved = False
        if np.linalg.norm(target_pos - self.target_init_pos) > 0.02:  # 阈值可调整
            print (f"target_pos: {target_pos}, target_init_pos: {self.target_init_pos}")
            reward -= 5
            target_moved = True

        print(f"Distance: {dist}, Reward: {reward}")
        self.prev_dist = dist
        terminated = dist < 0.02
        if terminated:
            print(f"terminated")
            reward += 10
        self.current_step += 1
        truncated = self.current_step >= self.max_steps or target_moved
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 返回观测向量
        right_pad_pos = self.data.xpos[self.right_pad_id]
        left_pad_pos = self.data.xpos[self.left_pad_id]
        middle = (right_pad_pos + left_pad_pos) / 2
        target_pos = self.data.xpos[self.target_body_id]
        return np.concatenate([middle, target_pos]).astype(np.float32)

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()