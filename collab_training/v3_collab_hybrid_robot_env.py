import gymnasium as gym
import numpy as np
import mujoco
from first_robot_controller.mirobot_controller import MirobotController
from config.env_config import Direction, Layer, FiniteState, RLRobotFiniteState
import concurrent.futures
from util_threads.object_placer import place_object_on_table, place_object_on_table_random
from util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter_with_flag
import threading
from stable_baselines3 import PPO
import random
import time
from enum import Enum
from utils.mujoco_object_color_randomiser import randomize_materials_at_runtime
from config.training_config import V1_MODEL_NAME


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1

class AgentRobot(Enum):
    ROBOT2 = 0
    ROBOT3 = 1

class ObjectColor:
    RED = np.array([1.0, 0.2, 0.2, 1.0])
    YELLOW = np.array([1.0, 1.0, 0.2, 1.0])
    
    @classmethod
    def identify(cls, color_array):
        if np.allclose(color_array, cls.RED):
            return "RED"
        elif np.allclose(color_array, cls.YELLOW):
            return "YELLOW"

class V3CollabHybridMuJoCoEnv(gym.Env):
    
    def _get_data_and_model(self):
        model = mujoco.MjModel.from_xml_path("../xml/collab_mirobot.xml")
        data = mujoco.MjData(model)
        time_step = 0.005
        model.opt.timestep = time_step  
        return model, data
    
    def __init__(self, action_repeat=1):
        super().__init__()
        self.action_repeat = action_repeat
        
        self.model, self.data = self._get_data_and_model()
        
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]
        
        # Robot 2 setup
        
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")

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
            "robot2:link6",          # arm end effector
            "robot2:vacuum_sphere"
        ]

        self.robot2_body_ids = []
        for body_name in self.robot2_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot2_body_ids.append(body_id)
            except:
                continue
            
        self.robot_2_origin_place_x_y = [-2, 1]
        self.robot2_recent_positions = []
        self.robot_2_target_position_x_y = None
        self.robot_2_random_picking_steps = None
        self.robot_2_random_picking_count = 0
        self.robot_2_random_placing_steps = None
        self.robot_2_random_placing_count = 0
        self.robot_2_target_placing_position = None
        self.robot_2_is_picking = False
        self.robot_2_is_placing = False
        self.robot_2_is_placing_upper = False
        self.robot_2_is_placing_lower = False
        self.robot_2_stopped = False
        self.robot_2_is_carrying_object = False
        self.robot_2_carrying_object_color = None
        self.robot_2_carrying_object_id = None
        self.check_robot_2_forbidden_collision_counter = 0
        self.robot_2_stop_wait_steps = 0
        

        # Robot2 setup end
        
        # Robot 3 setup

        self.robot_3_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot3:rover")

        self.robot3_bodies = [
            "robot3:rover",         # chassis
            "robot3:r-l-wheel",     # rear left wheel
            "robot3:r-r-wheel",     # rear right wheel  
            "robot3:f-l-wheel",     # front left wheel
            "robot3:f-l-wheel-hub", "robot3:f-l-wheel-1", "robot3:f-l-wheel-2",  # front left wheel hub and spokes
            "robot3:f-r-wheel-hub", "robot3:f-r-wheel-1", "robot3:f-r-wheel-2",  # front right wheel hub and spokes
            "robot3:f-r-wheel",     # front right wheel
            "robot3:base",          # arm base
            "robot3:base_link",     # arm base link
            "robot3:link1",         # arm joint 1
            "robot3:link2",         # arm joint 2
            "robot3:link3",         # arm joint 3
            "robot3:link4",         # arm joint 4
            "robot3:link5",         # arm joint 5
            "robot3:link6",          # arm end effector
            "robot3:vacuum_sphere"
        ]

        self.robot3_body_ids = []
        for body_name in self.robot3_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot3_body_ids.append(body_id)
            except:
                continue
            
        self.robot_3_origin_place_x_y = [-2, -1]
        self.robot3_recent_positions = []
        self.robot_3_target_position_x_y = None
        self.robot_3_random_picking_steps = None
        self.robot_3_random_picking_count = 0
        self.robot_3_random_placing_steps = None
        self.robot_3_random_placing_count = 0
        self.robot_3_target_placing_position = None
        self.robot_3_is_picking = False
        self.robot_3_is_placing = False
        self.robot_3_is_placing_upper = False
        self.robot_3_is_placing_lower = False
        self.robot_3_stopped = False
        self.robot_3_is_carrying_object = False
        self.robot_3_carrying_object_color = None
        self.robot_3_carrying_object_id = None
        self.check_robot_3_forbidden_collision_counter = 0
        self.robot_3_stop_wait_steps = 0

        # Robot3 setup end
        
        # General setup for RL robots
            
        self.prediction_steps = 5
        self.stop_wait_steps = 0
        self.required_stop_steps = 10
        
        self.forbidden_geoms = [
            "wall_front", 
            "wall_back", "wall_left", "wall_right",
            # "pickingplace:table0", "pickingplace:table2",
            "placingplace2:low_plane", "placingplace2:high_plane",
            "placingplace1:low_plane", "placingplace1:high_plane",  
        ]
        
        self.max_position = 3.0     
        self.max_speed = 2.0        
        self.max_distance = 8.0
        self.active_joint_id = None
        
        self.placingplace1_low_plane_pos = [2.8, 1, 0.23]
        self.placingplace2_low_plane_pos = [2.8, -1, 0.23]
        self.placingplace1_high_plane_pos = [2.8, 1, 0.43]
        self.placingplace2_high_plane_pos = [2.8, -1, 0.43]
        
        self.placingplace1_pos = [2.8, 1]
        self.placingplace2_pos = [2.8, -1]

        self.picking_positions = [
            [1, -2.45], # left
            [-1, -2.45],  # right
        ]

        self.placing_positions = [
            [2.80, -1],
            [2.80, 1]
        ]
        # General setup end
        
        # ACTIONS = {
        #     0: "drive 0",
        #     1: "drive 3",
        #     2: "drive -3",
        #     3: "steer 0",
        #     4: "steer -0.9",
        #     5: "steer 0.9",
        #     6: "pick",
        #     7: "place upper",
        #     8: "place lower"
        # }
        
        ACTIONS = {
            0: "do nothing",
            1: "forward",
            2: "backward",
            3: "left",
            4: "right",
            5: "do nothing",
            6: "pick",
            7: "place upper",
            8: "place lower"
        }
            
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        self.current_step = 0
        self.max_steps = 8000
        
        self.object_geoms = [
            "object0_geom", "object1_geom", "object2_geom", "object3_geom",
            "object4_geom", "object5_geom", "object6_geom", "object7_geom",
            "object8_geom", "object9_geom"
        ]
        
        self.object_body_names = [
            f"object{i}" for i in range(len(self.object_geoms))
        ]
        
        self.object_body_ids = []
        for body_name in self.object_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.object_body_ids.append(body_id)
            except:
                continue
        
        self.object_joints = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and joint_name.startswith("object") and joint_name.endswith(":joint"):
                try:
                    object_id = int(joint_name.split("object")[1].split(":")[0])
                    self.object_joints.append((object_id, i, joint_name))
                except (ValueError, IndexError):
                    continue

        self.object_joints.sort(key=lambda x: x[0])
        
        self.floor_body_name = ["floor"]   
        self.floor_body_ids = []
        for body_name in self.floor_body_name:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.floor_body_ids.append(body_id)
            except:
                continue 
            
        object_ids = self._get_object_ids(self.model)
        
        self.object_joint_ids = []
        
        for i in object_ids:
            joint_name = f"object{i}:joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.object_joint_ids.append((i, joint_id))
            
        self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": True}
            
        self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.remover_futures = []
        self.remover_shared_state = {"should_stop": False}
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids, self.remover_shared_state)

        self.agent_robot = random.choice(
            [AgentRobot.ROBOT2, AgentRobot.ROBOT3]
        )
        
        # self.rl_controlled_robot = None

        obs = self._get_obs(self.agent_robot)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        self.rl_model = PPO.load(V1_MODEL_NAME)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        randomize_materials_at_runtime(self.model)
        
        self.agent_robot = random.choice(
            [AgentRobot.ROBOT2, AgentRobot.ROBOT3]
        )
        
        if self.agent_robot == AgentRobot.ROBOT2:
            self.rl_controlled_robot = AgentRobot.ROBOT3
        if self.agent_robot == AgentRobot.ROBOT3:
            self.rl_controlled_robot = AgentRobot.ROBOT2
        
        self.current_step = 0
        
        self.robot_2_target_position_x_y = None
        self.robot_2_random_picking_steps = None
        self.robot_2_random_picking_count = 0
        self.robot_2_random_placing_steps = None
        self.robot_2_random_placing_count = 0
        self.robot_2_target_placing_position = None
        self.robot_2_is_picking = False
        self.robot_2_is_placing = False
        self.robot_2_stopped = False
        self.robot_2_is_carrying_object = False
        self.robot_3_carrying_object_id = None
        self.robot_2_stop_wait_steps = 0
        
        self.robot_3_target_position_x_y = None
        self.robot_3_random_picking_steps = None
        self.robot_3_random_picking_count = 0
        self.robot_3_random_placing_steps = None
        self.robot_3_random_placing_count = 0
        self.robot_3_target_placing_position = None
        self.robot_3_is_picking = False
        self.robot_3_is_placing = False
        self.robot_3_stopped = False
        self.robot_3_is_carrying_object = False
        self.robot_3_carrying_object_id = None
        self.robot_3_stop_wait_steps = 0
        
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        
        randomize_materials_at_runtime(self.model)
        
        if self.shared_state["stop"] is False:
            self.shared_state["stop"] = True
            self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": False}
        
        self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        
        self.remover_shared_state["should_stop"] = True
        self.remover_shared_state = {"should_stop": False}
        
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids, self.remover_shared_state)
        
        info = {}
        
        mujoco.mj_forward(self.model, self.data)
        
        max_wait_time = 5.0 
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            
            left_joint_id, _, right_joint_id, _ = self._get_placed_object_info()
            if left_joint_id is not None:
                self.robot_2_target_position_x_y = self.picking_positions[0]
                break
            elif right_joint_id is not None:
                self.robot_2_target_position_x_y = self.picking_positions[1]
                break
            else:
                mujoco.mj_forward(self.model, self.data)
                time.sleep(0.1)

        return self._get_obs(self.agent_robot), info
    
    def step(self, action):
        
        self._process_rl_robot()
        
        total_reward = 0
        terminated = False
        truncated = False
        final_obs = None
        final_info = {}
        
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._original_step(action)
            total_reward += reward
            final_obs = obs
            final_info.update(info)
            
            if terminated or truncated:
                break
        
        return final_obs, total_reward, terminated, truncated, final_info
        
    def _original_step(self, action):
        
        self._target_position_generator(self.agent_robot)
        
        break_flag = False
    
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) \
            and self._all_objects_removed():
            print("All objects have been placed and removed. Exiting")
            break_flag = True

        self.current_step += 1
        
        terminated = False
        truncated = False
        obs = self._get_obs(self.agent_robot)

        action_is_valid = self._check_action_validity(action, self.agent_robot)
        
        action_reward = 0
        
        if action_is_valid:
            action_reward = self._process_action(action, self.agent_robot)
            executed_action = action
        else:
            # print("Invalid action attempted, applying no-op instead.")
            action_reward = self._process_action(0, self.agent_robot) # Change action to no-op for logging purposes
            executed_action = 0
        
        
        # if self.current_step % 10 == 0:
        #     if action == 6:
                
        #         left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        #         if left_joint_id is not None:
        #             self.active_position = left_position
        #             self.active_joint_id = left_joint_id
        #             self.side = "left"
        #         elif right_joint_id is not None:
        #             self.active_position = right_position
        #             self.active_joint_id = right_joint_id
        #             self.side = "right"
        #         else:
        #             return False

        #         if self.agent_robot == AgentRobot.ROBOT2:
        #             robot_position = self.data.xpos[self.robot_2_rover_id][:2]
        #         elif self.agent_robot == AgentRobot.ROBOT3:
        #             robot_position = self.data.xpos[self.robot_3_rover_id][:2]

        #         distance_to_object = np.linalg.norm(self.active_position[:2] - robot_position)
                
        #         if distance_to_object < 0.4:
        #             print(f"Step: {self.current_step}, Agent Robot: {self.agent_robot.name}, distance to object: {distance_to_object:.3f}, object id: {self.active_joint_id}")
        #             print("action:", action, "executed_action:", executed_action)
        #             if self.agent_robot == AgentRobot.ROBOT2:
        #                 color = "blue"
        #             elif self.agent_robot == AgentRobot.ROBOT3:
        #                 color = "grey"
        #             print("color:", color)

        reward = self._reward_function_robot_2()
        reward += action_reward

        mujoco.mj_step(self.model, self.data)

        if self._check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 10
            terminated = True
            
        if self._check_robot_2_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 10
            terminated = True
            
        if self._check_robot_3_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 10
            terminated = True
            
        if break_flag:
            print("Task completed successfully! Terminating episode.")
            reward += 60
            terminated = True
        
        if self.current_step >= self.max_steps:
            print("Maximum steps reached, terminating episode.")
            terminated = True
            
        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("⚠️ QACC error detected, terminating episode.")
            truncated = True
            
        if self._check_floor_object_collision():
            print("Object-floor collision detected! Terminating episode.")
            truncated = True
        
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _reward_function_robot_2(self):
        reward = 0
        
        reward -= 0.001  # Small step penalty to encourage efficiency
        
        potential_field_reward = self._calculate_potential_field_reward() * 10
        # print("Potential field reward:", potential_field_reward)
        reward += potential_field_reward

        placement_penalty = self._check_placement_violations()
        reward += placement_penalty

        return reward
    
    def _process_action(self, action, agent_robot):
        action_reward = 0
        
        if action == 0:
            self.move_robot(0, 0, agent_robot)
        elif action == 1:
            self.move_robot(0.05, 0, agent_robot)
        elif action == 2:
            self.move_robot(-0.05, 0, agent_robot)
        elif action == 3:
            self.move_robot(0, 0.05, agent_robot)
        elif action == 4: 
            self.move_robot(0, -0.05, agent_robot)
        elif action == 5: 
            self.move_robot(0, 0, agent_robot)
        elif action == 6:
            print("Picking action executed")
            self._robot_picking(agent_robot)
            action_reward += 20
        elif action == 7:
            self._robot_placing_object(Layer.UPPER, agent_robot)
            action_reward += 20
        elif action == 8:
            self._robot_placing_object(Layer.LOWER, agent_robot)
            action_reward += 20

        return action_reward

    def move_robot(self, x_offset_value, y_offset_value, agent_robot):
        if agent_robot == AgentRobot.ROBOT2:
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
            qpos_adr = self.model.jnt_qposadr[rover_joint_id]
            
            current_pos = self.data.qpos[qpos_adr:qpos_adr+3].copy()
            current_pos[0] += x_offset_value  # x
            current_pos[1] += y_offset_value  # y
            self.data.qpos[qpos_adr:qpos_adr+3] = current_pos
            
            qvel_adr = self.model.jnt_dofadr[rover_joint_id]
            self.data.qvel[qvel_adr:qvel_adr+6] = 0.0
            
            if self.robot_2_is_carrying_object and self.robot_2_carrying_object_id is not None:
                target_position = self.data.xpos[self.robot_2_rover_id].copy()
                target_position[2] += 0.02
                self._move_object_to_position(self.robot_2_carrying_object_id, target_position)
            
        elif agent_robot == AgentRobot.ROBOT3:
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot3:centroid")
            qpos_adr = self.model.jnt_qposadr[rover_joint_id]
            
            current_pos = self.data.qpos[qpos_adr:qpos_adr+3].copy()
            current_pos[0] += x_offset_value
            current_pos[1] += y_offset_value
            self.data.qpos[qpos_adr:qpos_adr+3] = current_pos
            
            qvel_adr = self.model.jnt_dofadr[rover_joint_id]
            self.data.qvel[qvel_adr:qvel_adr+6] = 0.0
            
            if self.robot_3_is_carrying_object and self.robot_3_carrying_object_id is not None:
                target_position = self.data.xpos[self.robot_3_rover_id].copy()
                target_position[2] += 0.02
                self._move_object_to_position(self.robot_3_carrying_object_id, target_position)

    def _check_action_validity(self, action, agent_robot):
        if action in [0, 1, 2, 3, 4, 5]:
            if agent_robot == AgentRobot.ROBOT2:
                if self.robot_2_is_picking or self.robot_2_is_placing:
                    return False
            elif agent_robot == AgentRobot.ROBOT3:
                if self.robot_3_is_picking or self.robot_3_is_placing:
                    return False
        
        if action == 6:
            if agent_robot == AgentRobot.ROBOT2:
                if self.robot_2_is_carrying_object:
                    return False
            elif agent_robot == AgentRobot.ROBOT3:
                if self.robot_3_is_carrying_object:
                    return False
            
        if action == 6:        
            if self._can_robot_pick(agent_robot) is False:
                return False
            
        if self._can_robot_pick(agent_robot):
            if action == 4:
                return False
        
        if self._can_robot_place(agent_robot):
            if action == 1:
                return False 
            
        if action in [7, 8]:
            if agent_robot == AgentRobot.ROBOT2:
                if not self.robot_2_is_carrying_object:
                    return False
            elif agent_robot == AgentRobot.ROBOT3:
                if not self.robot_3_is_carrying_object:
                    return False
        
        if action in [7, 8]:
            if self._can_robot_place(agent_robot) is False:
                return False
            
        if self.robot_2_is_picking or self.robot_3_is_picking:
            if action != 6:
                return False
            
        if self.robot_2_is_placing_upper or self.robot_3_is_placing_upper:
            if action != 7:
                return False

        if self.robot_2_is_placing_lower or self.robot_3_is_placing_lower:
            if action != 8:
                return False

        if self.robot_2_is_picking or \
            self.robot_3_is_picking or \
            self.robot_2_is_placing or \
            self.robot_3_is_placing:
            if action in [0, 1, 2, 3, 4, 5]:
                return False
            
        if agent_robot == AgentRobot.ROBOT2:
            if self.robot_2_carrying_object_color == "RED" and self._near_to_placing_place(agent_robot, self.placingplace1_pos):
                if action in [7, 8]:
                    return False
            elif self.robot_2_carrying_object_color == "YELLOW" and self._near_to_placing_place(agent_robot, self.placingplace2_pos):
                if action in [7, 8]:
                    return False
        elif agent_robot == AgentRobot.ROBOT3:
            if self.robot_3_carrying_object_color == "RED" and self._near_to_placing_place(agent_robot, self.placingplace1_pos):
                if action in [7, 8]:
                    return False
            elif self.robot_3_carrying_object_color == "YELLOW" and self._near_to_placing_place(agent_robot, self.placingplace2_pos):
                if action in [7, 8]:
                    return False

        return True

    def _check_placement_violations(self):
        penalty = 0
        
        placingplace_object_numbers = self._get_object_number_on_each_placing_place()
        
        for i, object_count in enumerate(placingplace_object_numbers):
            if object_count > 1:
                excess_objects = object_count - 1
                violation_penalty = -0.0001 * excess_objects  
                penalty += violation_penalty
        
        return penalty

    def _forward_robot(self, value, agent_robot):
        if agent_robot == AgentRobot.ROBOT2:
            self.data.ctrl[
                0 :
                0 + 1
            ] = [value]
        elif agent_robot == AgentRobot.ROBOT3:
            self.data.ctrl[
                SECOND_ROBOT_ACTION_SPACE_LENGTH :
                SECOND_ROBOT_ACTION_SPACE_LENGTH + 1
            ] = [value]
    
    def _steer_robot(self, value, agent_robot):
        if agent_robot == AgentRobot.ROBOT2:
            self.data.ctrl[
                1 :
                1 + 1
            ] = [value]
        elif agent_robot == AgentRobot.ROBOT3:
            self.data.ctrl[
                SECOND_ROBOT_ACTION_SPACE_LENGTH + 1 :
                SECOND_ROBOT_ACTION_SPACE_LENGTH + 1 + 1
            ] = [value]
            
    def _robot_picking(self, agent_robot):
        # self._brake_robot2(agent_robot)

        if agent_robot == AgentRobot.ROBOT2:
            self.robot_2_stopped = True
            self.robot_2_is_picking = True
            
            if self.robot_2_stopped:
                self._robot_picking_execution(agent_robot)
            
        elif agent_robot == AgentRobot.ROBOT3:
            self.robot_3_stopped = True
            self.robot_3_is_picking = True
            
            if self.robot_3_stopped:
                self._robot_picking_execution(agent_robot)     
    
    def _brake_robot2(self, agent_robot):
        self._steer_robot(0, agent_robot)
        self._forward_robot(0, agent_robot)
        
        if self.agent_robot == AgentRobot.ROBOT2:
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
        elif self.agent_robot == AgentRobot.ROBOT3:
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot3:centroid")

        rover_qvel_start = self.model.jnt_dofadr[rover_joint_id]
        linear_vel = self.data.qvel[rover_qvel_start:rover_qvel_start+3]
        speed = np.linalg.norm(linear_vel)
        
        if self.agent_robot == AgentRobot.ROBOT2:
            if speed < 0.0001: 
                self.robot_2_stop_wait_steps += 1                
                if self.robot_2_stop_wait_steps >= self.required_stop_steps:
                    self.robot_2_stopped = True
            else:
                self.robot_2_stop_wait_steps = 0
        elif self.agent_robot == AgentRobot.ROBOT3:
            if speed < 0.0001: 
                self.robot_3_stop_wait_steps += 1                
                if self.robot_3_stop_wait_steps >= self.required_stop_steps:
                    self.robot_3_stopped = True
            else:
                self.robot_3_stop_wait_steps = 0
                
    def _robot_picking_execution(self, agent_robot):
        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        if left_joint_id is not None:
            self.active_joint_id = left_joint_id
        elif right_joint_id is not None:
            self.active_joint_id = right_joint_id
            
        if agent_robot == AgentRobot.ROBOT2:
            if self.robot_2_random_picking_steps is None:
                # self.robot_2_random_picking_steps = random.randint(4, 10)
                self.robot_2_random_picking_steps = random.randint(1, 1)
                
            self.robot_2_random_picking_count += 1
            
            if self.robot_2_random_picking_count >= self.robot_2_random_picking_steps: 
                target_position = self.data.xpos[self.robot_2_rover_id].copy()
                target_position[2] += 0.02
                self._move_object_to_position(self.active_joint_id, target_position)
                self.robot_2_is_carrying_object = True
                self.robot_2_carrying_object_color = self._get_object_color(self.active_joint_id)
                self.robot_2_carrying_object_id = self.active_joint_id
                self.robot_2_is_picking = False
                self.robot_2_stopped = False
                self.robot_2_random_picking_steps = None
                self.robot_2_random_picking_count = 0
        elif agent_robot == AgentRobot.ROBOT3:
            if self.robot_3_random_picking_steps is None:
                # self.robot_3_random_picking_steps = random.randint(4, 10)
                self.robot_3_random_picking_steps = random.randint(1, 1)
                
            self.robot_3_random_picking_count += 1
            
            if self.robot_3_random_picking_count >= self.robot_3_random_picking_steps: 
                target_position = self.data.xpos[self.robot_3_rover_id].copy()
                target_position[2] += 0.02
                self._move_object_to_position(self.active_joint_id, target_position)
                self.robot_3_is_carrying_object = True
                self.robot_3_carrying_object_color = self._get_object_color(self.active_joint_id)
                self.robot_3_carrying_object_id = self.active_joint_id
                self.robot_3_is_picking = False
                self.robot_3_stopped = False
                self.robot_3_random_picking_steps = None
                self.robot_3_random_picking_count = 0
    
    def _robot_placing_object(self, layer, agent_robot):
        # self._brake_robot2(agent_robot)
        print("Placing action executed")
        if agent_robot == AgentRobot.ROBOT2:
            self.robot_2_stopped = True
            self.robot_2_is_placing = True
            
            if layer == Layer.UPPER:
                self.robot_2_is_placing_upper = True
            else:
                self.robot_2_is_placing_lower = True
            
            if self.robot_2_stopped:
                self._robot_placing_execution(layer, agent_robot)
            
        elif agent_robot == AgentRobot.ROBOT3:
            self.robot_3_stopped = True
            self.robot_3_is_placing = True
            
            if layer == Layer.UPPER:
                self.robot_3_is_placing_upper = True
            else:
                self.robot_3_is_placing_lower = True
            
            if self.robot_3_stopped:
                self._robot_placing_execution(layer, agent_robot)
                
    def _robot_placing_execution(self, layer, agent_robot):
        if agent_robot == AgentRobot.ROBOT2:
        
            if self.robot_2_random_placing_steps is None:
                if layer == Layer.UPPER:
                    # self.robot_2_random_placing_steps = random.randint(6, 10)
                    self.robot_2_random_placing_steps = random.randint(1, 1)
                else:
                    # self.robot_2_random_placing_steps = random.randint(2, 4)
                    self.robot_2_random_placing_steps = random.randint(1, 1)

            self.robot_2_random_placing_count += 1
            
            if self.robot_2_target_placing_position is None:
                robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
                distance_to_placingplace1 = np.linalg.norm(robot2_pos - self.placingplace1_pos)
                distance_to_placingplace2 = np.linalg.norm(robot2_pos - self.placingplace2_pos)
                # compare distance to placing positions
                if distance_to_placingplace1 < distance_to_placingplace2:
                    if layer == Layer.UPPER:
                        self.robot_2_target_placing_position = self.placingplace1_high_plane_pos.copy()
                    else:
                        self.robot_2_target_placing_position = self.placingplace1_low_plane_pos.copy()
                else:
                    if layer == Layer.UPPER:
                        self.robot_2_target_placing_position = self.placingplace2_high_plane_pos.copy()
                    else:
                        self.robot_2_target_placing_position = self.placingplace2_low_plane_pos.copy()

            if self.robot_2_random_placing_count >= self.robot_2_random_placing_steps:
                
                # print(f"Placing object at {target_position}")
                self.robot_2_target_placing_position[2] += 0.02
                self._move_object_to_position(self.robot_2_carrying_object_id, self.robot_2_target_placing_position)
                self.robot_2_is_carrying_object = False
                self.robot_2_carrying_object_id = None
                self.robot_2_target_placing_position = None
                self.robot_2_is_placing = False
                self.robot_2_is_placing_lower = False
                self.robot_2_is_placing_upper = False
                self.robot_2_random_placing_steps = None
                self.robot_2_random_placing_count = 0
                self.robot_2_stop_wait_steps = 0
                
        elif agent_robot == AgentRobot.ROBOT3:
            if self.robot_3_random_placing_steps is None:
                if layer == Layer.UPPER:
                    # self.robot_3_random_placing_steps = random.randint(6, 10)
                    self.robot_3_random_placing_steps = random.randint(1, 1)
                else:
                    # self.robot_3_random_placing_steps = random.randint(2, 4)
                    self.robot_3_random_placing_steps = random.randint(1, 1)

            self.robot_3_random_placing_count += 1
            
            if self.robot_3_target_placing_position is None:
                robot3_pos = self.data.xpos[self.robot_3_rover_id][:2]
                distance_to_placingplace1 = np.linalg.norm(robot3_pos - self.placingplace1_pos)
                distance_to_placingplace2 = np.linalg.norm(robot3_pos - self.placingplace2_pos)
                # compare distance to placing positions
                if distance_to_placingplace1 < distance_to_placingplace2:
                    if layer == Layer.UPPER:
                        self.robot_3_target_placing_position = self.placingplace1_high_plane_pos.copy()
                    else:
                        self.robot_3_target_placing_position = self.placingplace1_low_plane_pos.copy()
                else:
                    if layer == Layer.UPPER:
                        self.robot_3_target_placing_position = self.placingplace2_high_plane_pos.copy()
                    else:
                        self.robot_3_target_placing_position = self.placingplace2_low_plane_pos.copy()

            if self.robot_3_random_placing_count >= self.robot_3_random_placing_steps:
                
                # print(f"Placing object at {target_position}")
                self.robot_3_target_placing_position[2] += 0.02
                self._move_object_to_position(self.robot_3_carrying_object_id, self.robot_3_target_placing_position)
                self.robot_3_is_carrying_object = False
                self.robot_3_carrying_object_id = None
                self.robot_3_target_placing_position = None
                self.robot_3_is_placing = False
                self.robot_3_is_placing_lower = False
                self.robot_3_is_placing_upper = False
                self.robot_3_random_placing_steps = None
                self.robot_3_random_placing_count = 0
                self.robot_3_stop_wait_steps = 0
      
    def _get_placed_object_info(self):
        left_joint_id = None
        left_position = None
        right_joint_id = None
        right_position = None
        
        for object_id, joint_id, joint_name in self.object_joints:
            body_id = self.model.jnt_bodyid[joint_id]
            position = self.data.xpos[body_id]

            if np.allclose(position, self.left_object_position, atol=0.1):
                left_joint_id = joint_id
                left_position = position.copy()

            elif np.allclose(position, self.right_object_position, atol=0.1):
                right_joint_id = joint_id
                right_position = position.copy()
        
        return left_joint_id, left_position, right_joint_id, right_position
    
    def _move_object_to_position(self, joint_id, new_position):
        qpos_adr = self.model.jnt_qposadr[joint_id]
        # joint_type = self.model.jnt_type[joint_id]
        
        # if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        self.data.qpos[qpos_adr:qpos_adr+3] = new_position 
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  
        dof_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[dof_adr:dof_adr+6] = 0.0


    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
    
    def _check_robot_robot_collision(self):
        """Directly detect collisions between two robots"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Detection logic
            is_robot1_involved = body1_id in self.robot3_body_ids or body2_id in self.robot3_body_ids
            is_robot2_involved = body1_id in self.robot2_body_ids or body2_id in self.robot2_body_ids
            
            if is_robot1_involved and is_robot2_involved:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                # print(f"🚨 ROBOT-ROBOT COLLISION: {geom1_name} <-> {geom2_name}")
                return True
        
        return False
    
    def _check_floor_object_collision(self):
        """Directly detect collisions between objects and the floor"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Detection logic
            is_object_involved = body1_id in self.object_body_ids or body2_id in self.object_body_ids
            is_floor_involved = body1_id in self.floor_body_ids or body2_id in self.floor_body_ids
            
            if is_object_involved and is_floor_involved:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                # print(f"🚨 OBJECT-FLOOR COLLISION: {geom1_name} <-> {geom2_name}")
                return True
        
        return False
    
    def _check_robot_2_forbidden_collision(self):
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
        
    def _check_robot_3_forbidden_collision(self):
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
            if ((body1_id in self.robot3_body_ids and geom2_name in self.forbidden_geoms) or
                (body2_id in self.robot3_body_ids and geom1_name in self.forbidden_geoms)):
                return True
        
        return False
    
    def _get_object_ids(self, model):
        object_ids = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("object") and name.endswith(":joint"):
                try:
                    num = int(name.split(":")[0][6:])  
                    object_ids.append(num)
                except Exception:
                    continue
        return sorted(object_ids)
    
    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        threading.Thread(
            target=place_object_on_table_random,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()

    def _start_object_remover_threads(self, model, data, object_joint_ids, remover_shared_state):
        
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=False) 
            except:
                pass
    
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.remover_futures = []
        
        # lower plane parameters
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23

        future1 = self.executor.submit(
            remove_object_on_plane_with_step_counter_with_flag,
            model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids, remover_shared_state,
            check_interval=0.05, min_delay_steps=1, max_delay_steps=2
        )

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        future2 = self.executor.submit(
            remove_object_on_plane_with_step_counter_with_flag,
            model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids, remover_shared_state,
            check_interval=0.05 , min_delay_steps=1, max_delay_steps=2
        )
        
    def _get_object_number_on_each_placing_place(self):
        
        def is_on_plane(obj_pos, plane_pos, plane_radius, plane_z, z_tol=0.05):
            dx = obj_pos[0] - plane_pos[0]
            dy = obj_pos[1] - plane_pos[1]
            dz = abs(obj_pos[2] - plane_z)
            return (dx**2 + dy**2) <= plane_radius**2 and dz < z_tol
        
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23
        
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43
        
        placingplace1_low_plane_object_number = 0
        placingplace2_low_plane_object_number = 0
        placingplace1_high_plane_object_number = 0
        placingplace2_high_plane_object_number = 0
        
        for i, joint_id in self.object_joint_ids:
            joint_name = f"object{i}:joint"
            qpos_adr = self.model.jnt_qposadr[joint_id]
            obj_pos = self.data.qpos[qpos_adr : qpos_adr+3]
            
            if is_on_plane(obj_pos, lower_plane_positions[0], lower_plane_radius, lower_plane_z):
                placingplace1_low_plane_object_number += 1
            
            elif is_on_plane(obj_pos, lower_plane_positions[1], lower_plane_radius, lower_plane_z):
                placingplace2_low_plane_object_number += 1

            elif is_on_plane(obj_pos, upper_plane_positions[0], upper_plane_radius, upper_plane_z):
                placingplace1_high_plane_object_number += 1

            elif is_on_plane(obj_pos, upper_plane_positions[1], upper_plane_radius, upper_plane_z):
                placingplace2_high_plane_object_number += 1

        return [placingplace1_low_plane_object_number, \
                placingplace2_low_plane_object_number, \
                placingplace1_high_plane_object_number, \
                placingplace2_high_plane_object_number]

    def _calculate_potential_field_reward(self):
        if self.agent_robot == AgentRobot.ROBOT2:
            robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
            attractive_potential = 0
            if self.robot_2_target_position_x_y is not None:
                target_pos = np.array(self.robot_2_target_position_x_y)
                distance_to_target = np.linalg.norm(target_pos - robot2_pos)
                attractive_potential = 0.5 * distance_to_target**2 

            repulsive_potential = 0
            robot3_pos = self.data.xpos[self.robot_3_rover_id][:2]
            distance_to_robot_3 = np.linalg.norm(robot2_pos - robot3_pos)
            # print(f"Distance to Robot 3: {distance_to_robot_3:.4f}")

            influence_distance = 1.0  # 影响范围
            if distance_to_robot_3 < influence_distance:
                repulsive_potential = 1.0 * (1/distance_to_robot_3 - 1/influence_distance)**2

            # 🔥 总势能
            total_potential = attractive_potential + repulsive_potential

            # 🔥 势能差作为奖励（鼓励势能降低）
            if hasattr(self, 'prev_potential'):
                potential_reward = self.prev_potential - total_potential
                # print(f"Potential Reward: {potential_reward:.4f}, Attractive: {attractive_potential:.4f}, Repulsive: {repulsive_potential:.4f}")
            else:
                # print("No previous potential, initializing potential reward to 0.")
                potential_reward = 0

            self.prev_potential = total_potential
            
        elif self.agent_robot == AgentRobot.ROBOT3:
            robot3_pos = self.data.xpos[self.robot_3_rover_id][:2]
            attractive_potential = 0
            if self.robot_3_target_position_x_y is not None:
                target_pos = np.array(self.robot_3_target_position_x_y)
                distance_to_target = np.linalg.norm(target_pos - robot3_pos)
                attractive_potential = 0.5 * distance_to_target**2

            repulsive_potential = 0
            robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
            distance_to_robot_2 = np.linalg.norm(robot3_pos - robot2_pos)
            # print(f"Distance to Robot 2: {distance_to_robot_2:.4f}")

            influence_distance = 1.0  # 影响范围
            if distance_to_robot_2 < influence_distance:
                repulsive_potential = 1.0 * (1/distance_to_robot_2 - 1/influence_distance)**2

            # 🔥 总势能
            total_potential = attractive_potential + repulsive_potential

            # 🔥 势能差作为奖励（鼓励势能降低）
            if hasattr(self, 'prev_potential'):
                potential_reward = self.prev_potential - total_potential
                # print(f"Potential Reward: {potential_reward:.4f}, Attractive: {attractive_potential:.4f}, Repulsive: {repulsive_potential:.4f}")
            else:
                # print("No previous potential, initializing potential reward to 0.")
                potential_reward = 0

            self.prev_potential = total_potential

        return potential_reward
    
    def _get_obs(self, agent_robot):
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        robot2_vel = self.data.cvel[self.robot_2_rover_id][:2]
        robot2_orientation = self._quaternion_to_yaw(self.data.xquat[self.robot_2_rover_id])
        # robot2_status = self.second_robot_status.value
        # robot_2_target_position = np.array(self.robot_2_target_position_x_y) 
        
        robot2_can_pick = self._can_robot_pick(agent_robot)
        robot2_can_place = self._can_robot_place(agent_robot)
        
        robot_2_is_carrying_object = 1.0 if self.robot_2_is_carrying_object else 0.0
        if self.robot_2_is_carrying_object:
            if self.robot_2_carrying_object_color == "RED":
                robot_2_carrying_object_color = 1.0
            elif self.robot_2_carrying_object_color == "YELLOW":
                robot_2_carrying_object_color = 0.0
        else:
            robot_2_carrying_object_color = -1

        robot_2_is_picking = 1.0 if self.robot_2_is_picking else 0.0
        robot_2_is_placing = 1.0 if self.robot_2_is_placing else 0.0
        robot_2_is_placing_upper = 1.0 if self.robot_2_is_placing_upper else 0.0
        robot_2_is_placing_lower = 1.0 if self.robot_2_is_placing_lower else 0.0

        robot3_pos = self.data.xpos[self.robot_3_rover_id][:2]
        robot3_vel = self.data.cvel[self.robot_3_rover_id][:2]
        robot3_orientation = self._quaternion_to_yaw(self.data.xquat[self.robot_3_rover_id])
        # robot3_status = self.third_robot_status.value
        # robot_3_target_position = np.array(self.robot_3_target_position_x_y)

        robot_3_can_pick = self._can_robot_pick(agent_robot)
        robot_3_can_place = self._can_robot_place(agent_robot)

        robot_3_is_carrying_object = 1.0 if self.robot_3_is_carrying_object else 0.0
        if self.robot_3_is_carrying_object:
            if self.robot_3_carrying_object_color == "RED":
                robot_3_carrying_object_color = 1.0
            elif self.robot_3_carrying_object_color == "YELLOW":
                robot_3_carrying_object_color = 0.0
        else:
            robot_3_carrying_object_color = -1
        
        robot_3_is_picking = 1.0 if self.robot_2_is_picking else 0.0
        robot_3_is_placing = 1.0 if self.robot_3_is_placing else 0.0
        robot_3_is_placing_upper = 1.0 if self.robot_3_is_placing_upper else 0.0
        robot_3_is_placing_lower = 1.0 if self.robot_3_is_placing_lower else 0.0
        
        placingplace_object_numbers = self._get_object_number_on_each_placing_place()
        placingplace_object_numbers_for_observation = np.zeros(4, dtype=int)
        for i, placingplace_object_number in enumerate(placingplace_object_numbers):
            if placingplace_object_number == 0:
                placingplace_object_numbers_for_observation[i] = 0
            elif placingplace_object_number == 1:
                placingplace_object_numbers_for_observation[i] = 1
            elif placingplace_object_number >= 1:
                placingplace_object_numbers_for_observation[i] = -1 # capacity exceeded

        if agent_robot == AgentRobot.ROBOT2:
            agent_robot_obs = np.concatenate([
                    robot2_pos / self.max_position,                    # [2] 位置
                    robot2_vel / self.max_speed,                       # [2] 速度 
                    [robot2_orientation / np.pi],                      # [1] 朝向
                    # [robot2_status / len(RLRobotFiniteState)],        # [1] 状态
                    # [robot_2_target_position[0] / self.max_position, robot_2_target_position[1] / self.max_position],  # [2] 目标位置
                    [robot2_can_pick],
                    [robot2_can_place],
                    [robot_2_is_carrying_object],                      # [1] 是否携带物体
                    [robot_2_carrying_object_color],                   # [1] 物体颜色
                    [robot_2_is_picking],                              # [1] 是否拾取
                    [robot_2_is_placing],                              # [1] 是否放置
                    [robot_2_is_placing_upper],                        # [1] 是否放置在上方
                    [robot_2_is_placing_lower],                        # [1] 是否放置在下方
                ], dtype=np.float32)

            another_robot_obs = np.concatenate([
                    robot3_pos / self.max_position,                    # [2] 位置
                    robot3_vel / self.max_speed,                       # [2] 速度 
                    [robot3_orientation / np.pi],                      # [1] 朝向
                    # [robot3_status / len(RLRobotFiniteState)],        # [1] 状态
                    # [robot_3_target_position[0] / self.max_position, robot_3_target_position[1] / self.max_position],  # [2] 目标位置
                    [robot_3_can_pick],
                    [robot_3_can_place],
                    [robot_3_is_carrying_object],                      # [1] 是否携带物体
                    [robot_3_carrying_object_color],                   # [1] 物体颜色
                    [robot_3_is_picking],                              # [1] 是否拾取
                    [robot_3_is_placing],                              # [1] 是否放置
                    [robot_3_is_placing_upper],                        # [1] 是否放置在上方
                    [robot_3_is_placing_lower],                        # [1] 是否放置在下方
            ], dtype=np.float32)
            
            target_obs = np.concatenate([
                self._get_target_relative_info(AgentRobot.ROBOT2),                  # [3] 目标相对位置+距离
                [1.0 if self.robot_2_target_position_x_y is not None else 0.0],  # [1] 是否有目标
            ], dtype=np.float32)
        
        elif agent_robot == AgentRobot.ROBOT3:
            
            agent_robot_obs = np.concatenate([
                    robot3_pos / self.max_position,                    # [2] 位置
                    robot3_vel / self.max_speed,                       # [2] 速度 
                    [robot3_orientation / np.pi],                      # [1] 朝向
                    # [robot3_status / len(RLRobotFiniteState)],        # [1] 状态
                    # [robot_3_target_position[0] / self.max_position, robot_3_target_position[1] / self.max_position],  # [2] 目标位置
                    [robot_3_can_pick],
                    [robot_3_can_place],
                    [robot_3_is_carrying_object],                      # [1] 是否携带物体
                    [robot_3_carrying_object_color],                   # [1] 物体颜色
                    [robot_3_is_picking],                              # [1] 是否拾取
                    [robot_3_is_placing],                              # [1] 是否放置
                    [robot_3_is_placing_upper],                        # [1] 是否放置在上方
                    [robot_3_is_placing_lower],                        # [1] 是否放置在下方
                ], dtype=np.float32)
            
            another_robot_obs = np.concatenate([
                robot2_pos / self.max_position,                    # [2] 位置
                robot2_vel / self.max_speed,                       # [2] 速度 
                [robot2_orientation / np.pi],                      # [1] 朝向
                # [robot2_status / len(RLRobotFiniteState)],        # [1] 状态
                # [robot_2_target_position[0] / self.max_position, robot_2_target_position[1] / self.max_position],  # [2] 目标位置
                [robot2_can_pick],
                [robot2_can_place],
                [robot_2_is_carrying_object],                      # [1] 是否携带物体
                [robot_2_carrying_object_color],                   # [1] 物体颜色
                [robot_2_is_picking],                              # [1] 是否拾取
                [robot_2_is_placing],                              # [1] 是否放置
                [robot_2_is_placing_upper],                        # [1] 是否放置在上方
                [robot_2_is_placing_lower],                        # [1] 是否放置在下方
            ], dtype=np.float32)
            
            target_obs = np.concatenate([
                self._get_target_relative_info(AgentRobot.ROBOT3),                  # [3] 目标相对位置+距离
                [1.0 if self.robot_3_target_position_x_y is not None else 0.0],  # [1] 是否有目标
            ], dtype=np.float32)
        
        
        walls = {"left": -3.0, "right": 3.0, "front": 3.0, "back": -3.0}
        if agent_robot == AgentRobot.ROBOT2:
            robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        elif agent_robot == AgentRobot.ROBOT3:
            robot2_pos = self.data.xpos[self.robot_3_rover_id][:2]
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],   
            walls["right"] - robot2_pos[0],  
            robot2_pos[1] - walls["back"],   
            walls["front"] - robot2_pos[1]   
        ])
    
        obs = np.concatenate([
            # 自身状态 [5维]
            agent_robot_obs,
            # 另一个机器人状态 [5维]
            another_robot_obs,
            # 目标信息 [4维]
            target_obs,
            
            # 放置位置状态 [4维]
            placingplace_object_numbers_for_observation,       # [4] 每个放置位置的物体数量
            
            wall_distances / self.max_distance,
            
        ], dtype=np.float32)
        
        return obs
    
    def _can_robot_pick(self, agent_robot):
        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        if left_joint_id is not None:
            self.active_position = left_position
            self.active_joint_id = left_joint_id
            self.side = "left"
        elif right_joint_id is not None:
            self.active_position = right_position
            self.active_joint_id = right_joint_id
            self.side = "right"
        else:
            return False

        if agent_robot == AgentRobot.ROBOT2:
            robot_position = self.data.xpos[self.robot_2_rover_id][:2]
        elif agent_robot == AgentRobot.ROBOT3:
            robot_position = self.data.xpos[self.robot_3_rover_id][:2]

        distance_to_object = np.linalg.norm(self.active_position[:2] - robot_position)
        
        if distance_to_object > 0.50:
            return False
        
        return True

    def _can_robot_place(self, agent_robot):
        if self._near_to_placing_place(agent_robot, self.placingplace1_pos ) or \
            self._near_to_placing_place(agent_robot, self.placingplace2_pos ):
            return True
        
        return False

    def _near_to_placing_place(self, agent_robot, placingplace_pos):
        if agent_robot == AgentRobot.ROBOT2:
            robot_position = self.data.xpos[self.robot_2_rover_id][:2]
        elif agent_robot == AgentRobot.ROBOT3:
            robot_position = self.data.xpos[self.robot_3_rover_id][:2]

        distance_to_placing_place = np.linalg.norm(placingplace_pos - robot_position)
        
        if placingplace_pos == self.placingplace1_pos:
            threshold = 1.0
        elif placingplace_pos == self.placingplace2_pos:
            threshold = 0.5

        if distance_to_placing_place > threshold:
            return False

        return True

    def _get_target_relative_info(self, robot_number):
        """获取目标相对信息"""
        if robot_number == AgentRobot.ROBOT2:
            robot_pos = self.data.xpos[self.robot_2_rover_id][:2]

            if self.robot_2_target_position_x_y is not None:
                target_pos = np.array(self.robot_2_target_position_x_y)
                target_rel = target_pos - robot_pos
                distance = np.linalg.norm(target_rel)
                return np.concatenate([
                    target_rel / self.max_distance,     # [2] 归一化相对位置
                    [distance / self.max_distance]      # [1] 归一化距离
                ])
            else:
                return np.zeros(3)
        elif robot_number == AgentRobot.ROBOT3:
            robot_pos = self.data.xpos[self.robot_3_rover_id][:2]

            if self.robot_3_target_position_x_y is not None:
                target_pos = np.array(self.robot_3_target_position_x_y)
                target_rel = target_pos - robot_pos
                distance = np.linalg.norm(target_rel)
                return np.concatenate([
                    target_rel / self.max_distance,     # [2] 归一化相对位置
                    [distance / self.max_distance]      # [1] 归一化距离
                ])
            else:
                return np.zeros(3)

    def _all_objects_removed(self):
        for _, joint_id, name in self.object_joints:
            body_id = self.model.jnt_bodyid[joint_id]
            position = self.data.xpos[body_id]
            position_x = position[0]
            if position_x > 3.1:  # x > 3.1 means removed
                continue
            else:
                return False
        return True

    def _get_object_color(self, joint_id):
        body_id = self.model.jnt_bodyid[joint_id]
        geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
        if geom_ids:
            color_array = self.model.geom_rgba[geom_ids[0]]
            color_name = ObjectColor.identify(color_array)
            return color_name
        
    def _target_position_generator(self, agent_robot):
        
        if agent_robot == AgentRobot.ROBOT2:
            
            robot_2_pos = self.data.xpos[self.robot_2_rover_id][:2]
            self.robot_2_target_position_x_y = robot_2_pos
            
            if not self.robot_2_is_carrying_object:
                left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

                if left_joint_id is not None:
                    self.active_position = left_position
                    self.active_joint_id = left_joint_id
                    self.side = "left"
                elif right_joint_id is not None:
                    self.active_position = right_position
                    self.active_joint_id = right_joint_id
                    self.side = "right"

                self.robot_2_target_position_x_y = self.active_position[:2]
                
            if self.robot_2_is_carrying_object:
                if self.robot_2_carrying_object_color == "YELLOW":
                    self.robot_2_target_position_x_y = [2.8, 1.0]
                elif self.robot_2_carrying_object_color == "RED":
                    self.robot_2_target_position_x_y = [2.8, -1.0]

            # print(f"Robot 2 Target Position: {self.robot_2_target_position_x_y}")

        elif agent_robot == AgentRobot.ROBOT3:
            robot_3_pos = self.data.xpos[self.robot_3_rover_id][:2]
            self.robot_3_target_position_x_y = robot_3_pos

            if not self.robot_3_is_carrying_object:
                left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

                if left_joint_id is not None:
                    self.active_position = left_position
                    self.active_joint_id = left_joint_id
                    self.side = "left"
                elif right_joint_id is not None:
                    self.active_position = right_position
                    self.active_joint_id = right_joint_id
                    self.side = "right"

                self.robot_3_target_position_x_y = self.active_position[:2]
                
            if self.robot_3_is_carrying_object:
                if self.robot_3_carrying_object_color == "YELLOW":
                    self.robot_3_target_position_x_y = [2.8, 1.0]
                elif self.robot_3_carrying_object_color == "RED":
                    self.robot_3_target_position_x_y = [2.8, -1.0]
                    
            # print(f"Robot 3 Target Position: {self.robot_3_target_position_x_y}")
    
    def _quaternion_to_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _process_rl_robot(self):
        obs = self._get_obs(self.rl_controlled_robot)

        self._target_position_generator(self.rl_controlled_robot)

        action, _ = self.rl_model.predict(obs, deterministic=False)
        
        action_is_valid = self._check_action_validity(action, self.rl_controlled_robot)
        
        if action_is_valid:
            action_reward = self._process_action(action, self.rl_controlled_robot)
        else:
            action_reward = self._process_action(0, self.rl_controlled_robot)