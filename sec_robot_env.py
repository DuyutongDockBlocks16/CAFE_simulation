import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController 
import threading
from object_remover import remove_object_on_plane
from object_placer import place_object_on_table
from env_config import FiniteState

ACTION_SPACE_REDUCTION = 13  # Number of actuators to be reduced from the action space

class SecondRobotMuJoCoEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        time_step = 0.005
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
        
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        # self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "placingplace1:low_plane")
        self.target_position_x_y = [2, -2] 

        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        num_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )
        self.max_steps = 100000
        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "pickingplace:table0", "pickingplace:table1", 
            "pickingplace:table2", "pickingplace:table3",
            "placingplace2:low_plane", "placingplace2:high_plane",
            "placingplace1:low_plane", "placingplace1:high_plane"
        ]

        robot_bodies = [
            "robot2:rover",         # chassis
            "robot2:r-l-wheel",     # rear left wheel
            "robot2:r-r-wheel",     # rear right wheel  
            "robot2:f-l-wheel",     # front left wheel
            "robot2:f-l-wheel-hub", "robot2:f-l-wheel-1", "robot2:f-l-wheel-2",  # front left wheel hub and spokes
            "robot2:f-r-wheel-hub", "robot2:f-r-wheel-1", "robot2:f-r-wheel-2",  # front right wheel hub and spokes
            "robot2:f-r-wheel",     # front right wheel
            "robot2:link2",         # arm joint 2
            "robot2:link3",         # arm joint 3
            "robot2:link4",         # arm joint 4
            "robot2:link5",         # arm joint 5
            "robot2:link6"          # arm end effector
        ]

        self.robot_body_ids = []
        for body_name in robot_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot_body_ids.append(body_id)
            except:
                continue
        
        self.robot1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        self.safe_distance = 0.8
        self.prev_dist = None

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
        self.first_robot_controller.reset_all_joints()

        mujoco.mj_forward(self.model, self.data)
        # self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False}
        # self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)

        self.current_step = 0

        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id]
        self.prev_dist = np.linalg.norm(robot_2_rover_pos[0:2] - self.target_position_x_y)

        return self._get_obs(), {}

    def step(self, action):  
        terminated = False
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(action)] = action
        status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            terminated = True

        mujoco.mj_step(self.model, self.data)

        robot_2_rover_pos = self._get_obs()

        reward, reached, crushed = self.reward_function(robot_2_rover_pos)

        if self.check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 50
            terminated = True

        if crushed:
            print("Robot distance violation detected! Terminating episode.")
            terminated = True

        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("QACC error detected! Simulation unstable, exiting loop.")
            terminated = True

        if reached:
            terminated = True

        self.current_step += 1

        truncated = self.current_step >= self.max_steps
        info = {}
        return robot_2_rover_pos, reward, terminated, truncated, info

    def _get_obs(self):
        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id]
        # target_area_pos = self.model.geom_pos[self.target_area_geom_id]
        # return np.concatenate([robot_2_rover_pos]).astype(np.float32)
        return robot_2_rover_pos[0:2]

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
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

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

    def check_robot_forbidden_collision(self):
        # Check all contact points
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body corresponding to geom
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Get geom names
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            # Check if robot body collides with forbidden geom
            if ((body1_id in self.robot_body_ids and geom2_name in self.forbidden_geoms) or
                (body2_id in self.robot_body_ids and geom1_name in self.forbidden_geoms)):
                return True
        
        return False

    def check_robot_distance_violation(self):
        """Check if the distance between two robot chassis is less than the safe distance"""
        
        # Get positions of both robot chassis
        robot1_pos = self.data.xpos[self.robot1_rover_id]
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        
        return distance < self.safe_distance

    def reward_function(self, robot_2_rover_pos):
        dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
        target_reward = (self.prev_dist - dist_to_target) * 50

        robot_distance = self.get_robot_distance()
        safety_reward = self.calculate_safety_reward(robot_distance)

        crushed = False
        if safety_reward == -50 :
            crushed = True

        time_penalty = -0.1

        arrival_bonus = 0

        reached = dist_to_target < 0.01
        if reached:
            arrival_bonus = 200

        total_reward = target_reward + safety_reward + time_penalty + arrival_bonus

        self.prev_dist = dist_to_target

        return total_reward, reached, crushed

    def get_robot_distance(self):
        """Calculate the distance between the two robots"""
        robot1_pos = self.data.xpos[self.robot1_rover_id]
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        return np.linalg.norm(robot1_pos - robot2_pos)

    def calculate_safety_reward(self, robot_distance):
        """Safety distance reward design"""
        if robot_distance < 0.8:     # Collision
            return -50             # Large penalty
        elif robot_distance < 1.2:   # Danger zone
            return -20             # Medium penalty  
        elif robot_distance < 1.6:   # Warning zone
            return -5               # Small penalty
        else:                        # Safe zone
            return 2       