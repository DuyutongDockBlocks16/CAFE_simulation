import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController 
import threading
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
from config.env_config import FiniteState

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

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        
        # self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "placingplace1:low_plane")
        self.target_position_x_y = [2, -2] 

        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        num_actuators = self.model.nu

        self.low_bounds = np.array([-3.0, -0.9], dtype=np.float32)
        self.high_bounds = np.array([3.0, 0.9], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )

        self.max_steps = 8000
        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "pickingplace:table0", "pickingplace:table1", 
            "pickingplace:table2", "pickingplace:table3",
            "placingplace2:low_plane", "placingplace2:high_plane",
            "placingplace1:low_plane", "placingplace1:high_plane",  
        ]

        self.robot1_bodies = [
            "robot1:rover",         # chassis
            "robot1:r-l-wheel",     # rear left wheel
            "robot1:r-r-wheel",     # rear right wheel  
            "robot1:f-l-wheel",     # front left wheel
            "robot1:f-l-wheel-1", "robot1:f-l-wheel-2",  # front left wheel spokes
            "robot1:f-r-wheel",     # front right wheel
            "robot1:f-r-wheel-1", "robot1:f-r-wheel-2",  # front right wheel spokes
            "robot1:base",          # arm base
            "robot1:base_link",     # arm base link
            "robot1:link1",         # arm joint 1
            "robot1:link2",         # arm joint 2
            "robot1:link3",         # arm joint 3
            "robot1:link4",         # arm joint 4
            "robot1:link5",         # arm joint 5
            "robot1:link6",         # arm end effector
            "vacuum_sphere"         # vacuum gripper
        ]

        self.robot2_bodies = [
            "robot2:rover",         # chassis
            "robot2:r-l-wheel",     # rear left wheel
            "robot2:r-r-wheel",     # rear right wheel  
            "robot2:f-l-wheel",     # front left wheel
            "robot2:f-l-wheel-hub", "robot2:f-l-wheel-1", "robot2:f-l-wheel-2",  # front left wheel hub and spokes
            "robot2:f-r-wheel-hub", "robot2:f-r-wheel-1", "robot2:f-r-wheel-2",  # front right wheel hub and spokes
            "robot2:f-r-wheel",     # front right wheel
            "robot2:base",          # arm base
            "robot2:base_link",     # arm base link
            "robot2:link1",         # arm joint 1
            "robot2:link2",         # arm joint 2
            "robot2:link3",         # arm joint 3
            "robot2:link4",         # arm joint 4
            "robot2:link5",         # arm joint 5
            "robot2:link6"          # arm end effector
        ]

        self.robot1_body_ids = []
        for body_name in self.robot1_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot1_body_ids.append(body_id)
            except:
                continue

        self.robot2_body_ids = []
        for body_name in self.robot2_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot2_body_ids.append(body_id)
            except:
                continue
        
        self.robot1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        self.safe_distance = 0.8
        self.prev_dist = None
        self.init_dist = None
        self.static_counter = 0
        self.max_static_steps = 400 

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
        self.init_dist = self.prev_dist

        self.prev_position = None
        self.static_counter = 0

        return self._get_obs(), {}

    def step(self, action):  
        normalized_action = np.clip(action, -1, 1)
        real_action = self.low_bounds + (normalized_action + 1) * (self.high_bounds - self.low_bounds) / 2

        terminated = False
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            terminated = True

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]  # Get only the first two coordinates (x, y)
        # print(robot_2_rover_pos)
        reward, reached = self.reward_function(robot_2_rover_pos)

        static_penalty = self.calculate_static_penalty(action, robot_2_rover_pos)
        reward += static_penalty

        truncated = False
        if self.check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 20000
            truncated = True

        if self.check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 20000
            truncated = True

        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("QACC error detected! Simulation unstable, exiting loop.")
            truncated = True

        if reached:
            print("Robot2 has reached the target area! Terminating episode.")
            terminated = True

        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 10000
        
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]  
        # robot2_vel = self.data.qvel[:2]  
        
        target_pos = np.array(self.target_position_x_y) 
        target_rel = target_pos - robot2_pos  
        target_distance = np.linalg.norm(target_rel) 
        target_angle = np.arctan2(target_rel[1], target_rel[0]) 

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])  

        walls = {
            "left": -3.0,   
            "right": 3.0,   
            "front": 3.0,   
            "back": -3.0    
        }
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],   
            walls["right"] - robot2_pos[0],  
            robot2_pos[1] - walls["back"],   
            walls["front"] - robot2_pos[1]   
        ])

        placing_place_1_pos = np.array([2.8, 1.0])  
        placing_1_rel = placing_place_1_pos - robot2_pos
        placing_1_distance = np.linalg.norm(placing_1_rel)
        # placing_1_angle = np.arctan2(placing_1_rel[1], placing_1_rel[0])

        placing_place_2_pos = np.array([2.8, -1.0])
        placing_2_rel = placing_place_2_pos - robot2_pos
        placing_2_distance = np.linalg.norm(placing_2_rel)
        # placing_2_angle = np.arctan2(placing_2_rel[1], placing_2_rel[0])

        
        max_position = 3.0     
        max_speed = 2.0        
        max_distance = 9     
        
        observation = np.concatenate([
            robot2_pos / max_position,                
            # robot2_vel / max_speed,                   
            
            target_pos / max_position,               
            [target_distance / max_distance,           
            target_angle / np.pi],                   
            

            robot1_pos / max_position,               
            [robot1_distance / max_distance,          
            robot1_angle / np.pi],                 
            
            wall_distances / max_distance,            

            placing_place_1_pos / max_position,       
            [placing_1_distance / max_distance],      
            
            placing_place_2_pos / max_position,       
            [placing_2_distance / max_distance]       
        ], dtype=np.float32)

        return observation

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
            if ((body1_id in self.robot2_body_ids and geom2_name in self.forbidden_geoms) or
                (body2_id in self.robot2_body_ids and geom1_name in self.forbidden_geoms)):
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
        # target_reward = -(abs(robot_2_rover_pos[0] - self.target_position_x_y[0]) + abs(robot_2_rover_pos[1] - self.target_position_x_y[1]))
        # target_reward = (self.prev_dist - dist_to_target) * 100
        # if target_reward < 0:
        #     target_reward -= 0.01 * self.current_step / 1000 
        # else:
        #     target_reward += 0.01 * self.current_step / 1000
        

        progress_reward = 0
        if self.prev_dist is not None:
            progress_amount = (self.prev_dist - dist_to_target) * 100
            
            
            if dist_to_target > 4.0:
                coefficient = 2.0      
            elif dist_to_target > 3.0:
                coefficient = 3.0      
            elif dist_to_target > 2.0:
                coefficient = 4.0     
            elif dist_to_target > 1.0:
                coefficient = 5.0      
            elif dist_to_target > 0.5:
                coefficient = 6.0      
            elif dist_to_target > 0.2:
                coefficient = 8.0      
            else:
                coefficient = 10.0     # 保持10.0
            
            progress = progress_amount * coefficient
            
            if progress > 0:
                progress_reward = progress
            else:
                progress_reward = progress * 0.5


        robot_distance = self.get_robot_distance()
        safety_reward = self.calculate_safety_reward(robot_distance)

        # crushed = False
        # if safety_reward == -5000 :
        #     crushed = True

        time_penalty = -0.3

        arrival_bonus = 0

        reached = dist_to_target < 0.1
        if reached:
            arrival_bonus = 200000

        total_reward = progress_reward + safety_reward + arrival_bonus + time_penalty
        # total_reward = target_reward + time_penalty + arrival_bonus

        # print(f"Robot2 Position: {robot_2_rover_pos}, Distance to Target: {dist_to_target:.2f}, ")

        self.prev_dist = dist_to_target

        return total_reward, reached

    def get_robot_distance(self):
        """Calculate the distance between the two robots"""
        robot1_pos = self.data.xpos[self.robot1_rover_id]
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        return np.linalg.norm(robot1_pos - robot2_pos)

    def calculate_safety_reward(self, robot_distance):
        """Safety distance reward design"""
        if robot_distance < 0.8:     # Collision
            return 0            # Large penalty
        elif robot_distance < 1.0:   # Danger zone
            return 0.5             # Medium penalty  
        # elif robot_distance < 2.0: 
        #     return 1               
        else:                        # Safe zone
            return 1

    def check_robot_robot_collision(self):
        """Directly detect collisions between two robots"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Detection logic
            is_robot1_involved = body1_id in self.robot1_body_ids or body2_id in self.robot1_body_ids
            is_robot2_involved = body1_id in self.robot2_body_ids or body2_id in self.robot2_body_ids
            
            if is_robot1_involved and is_robot2_involved:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                # print(f"🚨 ROBOT-ROBOT COLLISION: {geom1_name} <-> {geom2_name}")
                return True
        
        return False

    def calculate_static_penalty(self, action, current_pos):
        penalty = 0
        

        if self.prev_position is not None:
            position_change = np.linalg.norm(current_pos - self.prev_position)
            
            if position_change < 0.05:  
                self.static_counter += 1
                if self.static_counter > 50:  
                    
                    penalty -= min((self.static_counter - 50) * 0.05, -5) 
            else:
                self.static_counter = max(0, self.static_counter - 3)  
                penalty += 0.5 
        
        self.prev_position = current_pos.copy()
        
        return penalty