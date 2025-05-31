import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController 
import threading
from object_remover import remove_object_on_plane
from object_placer import place_object_on_table
from env_config import FiniteState

class SecondRobotMuJoCoEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        time_step = 0.001
        self.model.opt.timestep = time_step

        object_ids = self.get_object_ids(self.model)
        self.object_joint_ids = []
        for i in object_ids:
            joint_name = f"object{i}:joint"
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.object_joint_ids.append((i, joint_id))
            except Exception:
                print(f"Joint {joint_name} not found in main thread")
        
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]

        mujoco.mj_forward(self.model, self.data)

        self.first_robot_controller = MirobotController(self.model, self.data, self.left_object_position, self.right_object_position)

        # Start the asynchronous thread
        self.start_object_remover_threads(self.model, self.data, self.object_joint_ids)

        self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}
        
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "placingplace1:low_plane")

        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        num_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(num_actuators - 7,), 
            dtype=np.float32
        )
        self.max_steps = 1000000
        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

    def reset(self, seed=None, options=None):

        if self.shared_state["stop"] is False:
            self.shared_state["stop"] = True
            while self.shared_state["stopped"] is False:
                pass
            self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": False}

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl

        self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)

        self.first_robot_controller.set_state(FiniteState.IDLE)

        mujoco.mj_forward(self.model, self.data)
        # self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False}
        # self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)

        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):  
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        self.data.ctrl[7:7+len(action)] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = 0.0 
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        target_obj_pos = self.data.xpos[self.target_body_id]
        target_area_pos = self.model.geom_pos[self.target_area_geom_id]
        return np.concatenate([target_obj_pos, target_area_pos]).astype(np.float32)

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()


    def get_object_ids(self, model):
        object_ids = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("object") and name.endswith(":joint"):
                # extract the N from the name
                try:
                    num = int(name.split(":")[0][6:])  # "objectN:joint" -> N
                    object_ids.append(num)
                except Exception:
                    continue
        return sorted(object_ids)

    def start_object_remover_threads(self, model, data, object_joint_ids):
        # lower plane parameters
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23

        threading.Thread(
            target=remove_object_on_plane,
            args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
            daemon=True
        ).start()

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.08
        upper_plane_z = 0.33

        threading.Thread(
            target=remove_object_on_plane,
            args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
            daemon=True
        ).start()


    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        # object positions parameters
        threading.Thread(
            target=place_object_on_table,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()
