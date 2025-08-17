import gymnasium as gym
from simplized_hybrid_controller import SimpleHybridController
import numpy as np
import mujoco
from mirobot_controller import MirobotController
from config.env_config import Direction, Layer, FiniteState, RLRobotFiniteState
import concurrent.futures
from util_threads.object_placer import place_object_on_table
from util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter_with_flag
import threading
from stable_baselines3 import PPO
import random
import time
from enum import Enum

FIRST_ROBOT_ACTION_SPACE_LENGTH = 8
SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH = 2
SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH = 6

class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1

class FsmHybridMuJoCoEnv(gym.Env):
    
    def _get_data_and_model(self):
        model = mujoco.MjModel.from_xml_path("xml/scene_mirobot.xml")
        data = mujoco.MjData(model)
        time_step = 0.005
        model.opt.timestep = time_step  
        return model, data
    
    def __init__(self, action_repeat=40):
        super().__init__()
        self.action_repeat = action_repeat
        
        sec_robot_forward_model_path = "models/final_model_continued_21600K_20250707_165449.zip.bak"
        sec_robot_backward_model_path="models/final_model_continued_56000K_20250724_140118.zip.bak"

        self.sec_robot_navigation_forward_model = PPO.load(sec_robot_forward_model_path)
        self.sec_robot_navigation_backward_model = PPO.load(sec_robot_backward_model_path)
        
        self.model, self.data = self._get_data_and_model()
        
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]
        
        self.first_robot_controller = MirobotController(self.model, self.data, self.left_object_position, self.right_object_position)
        
        # Robot 1 setup
        
        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        
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
            "robot1:vacuum_sphere"  # vacuum gripper
        ]

        self.robot1_body_ids = []
        for body_name in self.robot1_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot1_body_ids.append(body_id)
            except:
                continue
            
        self.robot1_recent_positions = []
        
        self.first_robot_status = FiniteState.IDLE
        
        self.first_robot_is_carrying = False
        
        # Robot 1 setup end
        
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
        
        self.second_robot_status = RLRobotFiniteState.IDLE
        
        self.robot_2_target_position_x_y = None
        
        self.robot_2_random_picking_steps = None
        self.robot_2_random_picking_count = 0
        
        self.robot_2_random_placing_steps = None
        self.robot_2_random_placing_count = 0
        
        self.robot_2_target_placing_position = None
        
        self.second_robot_is_picking = False

        # Robot2 setup end
        
        # General setup for RL robots
            
        self.prediction_steps = 5
        self.stop_wait_steps = 0
        self.required_stop_steps = 50
        
        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "pickingplace:table0", "pickingplace:table2",
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
        
        # RL FSM:
        # IDLE = 0
        # NAVIGATE_TO_PICKING_POSITION = 1
        # PICKING_OBJECT = 2
        # NAVIGATE_TO_PLACING_POSITION = 3
        # PLACING_OBJECT = 4
        # MOVING_TO_ORIGIN_POSITION = 5
        # WAIT_FOR_FINISH = 6
        
        # action space:
        # 0-8 for Robot 2 actions
        ACTIONS = {
            # 0 Brake and wait
            # 1 Keep moving
            # 2 Move to pickingplace:table0
            # 3 Move to pickingplace:table1
            # 4 Pick
            # 5 Move to placingplace:table0
            # 6 Move to placingplace:table1
            # 7 Place Upper
            # 8 Place Lower
            0: "Brake and wait",             
            1: "Keep moving",     
            2: "Move to pickingplace",
            3: "Pick",         
            4: "Move to placingplace:table0",         
            5: "Move to placingplace:table1",   
            6: "Place Upper",   
            7: "Place Lower",  
            8: "Moving to origin position",  # This action is used to move the robot back to the origin position
            9: "Back car to adjust position",
            10: "Forward car to adjust position"
        }
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        
        # self.ALLOWED_ACTIONS = {
        #     RLRobotFiniteState.IDLE:                           [0, 2, 9, 10],
        #     RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:   [0, 2, 9, 10],
        #     RLRobotFiniteState.PICKING_OBJECT:                 [3],
        #     RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION:   [0, 4, 5, 9, 10],
        #     RLRobotFiniteState.PLACING_OBJECT:                 [0, 6, 7],
        #     RLRobotFiniteState.PLACING_OBJECT_UPPER:           [6],
        #     RLRobotFiniteState.PLACING_OBJECT_LOWER:           [7],
        #     RLRobotFiniteState.MOVING_TO_ORIGIN_POSITION:      [0, 8, 9, 10],
        #     RLRobotFiniteState.WAIT_FOR_FINISH:                [0]
        # }
        
        self.ALLOWED_ACTIONS = {
            RLRobotFiniteState.IDLE:                           [0, 2],
            # RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:   [0, 2, 10],
            RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:   [0, 2],
            RLRobotFiniteState.PICKING_OBJECT:                 [3],
            RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION: [0, 4, 5],
            # RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION:   [0, 4, 5],
            # RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1:   [0, 4, 9],
            RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1:   [0, 4],
            # RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2:   [0, 5, 9],
            RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2:   [0, 5],
            RLRobotFiniteState.PLACING_OBJECT:                 [0, 6, 7],
            RLRobotFiniteState.PLACING_OBJECT_UPPER:           [6],
            RLRobotFiniteState.PLACING_OBJECT_LOWER:           [7],
            RLRobotFiniteState.MOVING_TO_ORIGIN_POSITION:      [0, 8],
            RLRobotFiniteState.WAIT_FOR_FINISH:                [0]
        }

        self.current_step = 0
        self.max_steps = 500000
        
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

        obs = self._get_obs()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
    def get_action_mask(self):
        masks = np.zeros(self.action_space.n, dtype=bool)
        allowed = self.ALLOWED_ACTIONS[self.second_robot_status]
        masks[allowed] = True
        return masks
    
    def action_masks(self):
        """MaskablePPO期望的函数名"""
        return self.get_action_mask()  # 🔥 调用原函数
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        self.robot1_recent_positions = []
        self.robot2_recent_positions = []
        
        self.first_robot_status = RLRobotFiniteState.IDLE
        self.first_robot_controller.reset_all_joints()
        self.first_robot_controller.set_state(FiniteState.IDLE)
        self.first_robot_is_carrying = False
        
        self.second_robot_status = RLRobotFiniteState.IDLE
        
        self.robot_2_target_position_x_y = None
        
        self.robot_2_random_picking_steps = None
        self.robot_2_random_picking_count = 0
        
        self.robot_2_random_placing_steps = None
        self.robot_2_random_placing_count = 0
        
        self.robot_2_target_placing_position = None
        
        self.second_robot_is_picking = False
        
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        
        if self.shared_state["stop"] is False:
            self.shared_state["stop"] = True
            self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": False}
        
        self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        
        self.remover_shared_state["should_stop"] = True
        self.remover_shared_state = {"should_stop": False}
        
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids, self.remover_shared_state)
        
        info = {}
        info["action_mask"] = self.action_masks()
        
        mujoco.mj_forward(self.model, self.data)
        
        max_wait_time = 5.0  # 最多等待5秒
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
                print(f"⏳ 等待物体放置... (已等待 {time.time() - start_time:.1f}s)")
                time.sleep(0.1)

        return self._get_obs(), info

    def _get_obs(self):  
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]  
        
        try:
            robot2_body_id = self.robot_2_rover_id
            robot2_vel = self.data.cvel[robot2_body_id][:2]  
        except:
            robot2_vel = np.zeros(2)
        
        robot2_quat = self.data.xquat[self.robot_2_rover_id]
        robot2_orientation = self._quaternion_to_yaw(robot2_quat)

        # robot1_predicted_trajectory = self._predict_robot1_trajectory()
        # collision_risks, min_distances = self._calculate_collision_risk_timeline(robot2_pos, robot1_predicted_trajectory)
        
        # trajectory_features = self._extract_trajectory_features(robot1_predicted_trajectory, robot2_pos)

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])
        robot1_relative_angle = robot1_angle - robot2_orientation
        robot1_relative_angle = np.arctan2(np.sin(robot1_relative_angle), np.cos(robot1_relative_angle))
        robot1_rel_normalized = robot1_rel / self.max_distance

        placingplace_object_numbers = self._get_object_number_on_each_placing_place()
        
        placingplace_object_numbers_for_observation = np.zeros(4, dtype=int)
        for i, placingplace_object_number in enumerate(placingplace_object_numbers):
            if placingplace_object_number == 0:
                placingplace_object_numbers_for_observation[i] = 0
            elif placingplace_object_number == 1:
                placingplace_object_numbers_for_observation[i] = 1
            elif placingplace_object_number >= 1:
                placingplace_object_numbers_for_observation[i] = -1 # capacity exceeded
        
        observation = np.concatenate([
            # 🎯 机器人自身状态
            robot2_pos / self.max_position,                    # [2] 绝对位置
            robot2_vel / self.max_speed,                       # [2] 速度 
            [robot2_orientation / np.pi],                 # [1] 朝向 
            
            # 机器人2状态
            [self.second_robot_status.value / len(RLRobotFiniteState)],  # [1] 状态（归一化）
            # 机器人1状态
            [self.first_robot_status.value / len(FiniteState)],  # [1] 状态（归一化）
            
            # 🎯 机器人1相对信息（增强版）
            robot1_rel_normalized,                        # [2] 机器人1相对位置 
            [robot1_distance / self.max_distance],             # [1] 机器人1距离
            [robot1_relative_angle / np.pi],              # [1] 机器人1相对角度 

            # trajectory_features,

            # [np.max(collision_risks),                     # 最大风险
            #  np.mean(collision_risks),                    # 平均风险
            #  np.min(min_distances),                       # 最小距离
            #  np.argmax(collision_risks) / self.prediction_steps],  # 最危险时刻(归一化)
            
            placingplace_object_numbers_for_observation,  # [4] 每个放置位置的物体数量（归一化）
            
        ], dtype=np.float32)

        return observation
    
    def step(self, action):
        self._update_robot_tracking()
        
        # 如果action_repeat>1，重复执行action
        total_reward = 0
        terminated = False
        truncated = False
        final_obs = None
        final_info = {}
        
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info, action_switch = self._original_step(action)
            total_reward += reward
            final_obs = obs
            final_info.update(info)
            
            if terminated or truncated:
                break
            if action_switch:
                break
        
        return final_obs, total_reward, terminated, truncated, final_info
        
    def _original_step(self, action):
        
        break_flag = False
        
        self.first_robot_status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) \
            and self.first_robot_status == FiniteState.IDLE\
            and self.second_robot_status == RLRobotFiniteState.WAIT_FOR_FINISH:
            print("All objects have been placed. Exit")
            break_flag = True
            
        self.first_robot_controller.step(self.shared_state["current_object_position"])

        self.current_step += 1
        
        terminated = False
        truncated = False
        obs = self._get_obs()
        
        reward, action_switch = self._reward_function_robot_2(action)
        
        mujoco.mj_step(self.model, self.data)
        
        if self._check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 4000
            terminated = True
            
        if self._check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 4000
            terminated = True
            
        if break_flag:
            print("Task completed successfully! Terminating episode.")
            reward += 6000
            terminated = True
        
        if self.current_step >= self.max_steps:
            terminated = True
            
        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("⚠️ QACC error detected, terminating episode.")
            truncated = True
            
        if self._check_floor_object_collision():
            print("Object-floor collision detected! Terminating episode.")
            truncated = True
        
        info = {}
        info["action_mask"] = self.action_masks()
        
        return obs, reward, terminated, truncated, info, action_switch
    
    def _reward_function_robot_2(self, action):
        action_switch = False
        
        reward = 0.0
        
        time_penalty = 0.1
        reward -= time_penalty
        
        self.first_robot_is_carrying = False

        if self.first_robot_status not in [
            FiniteState.IDLE,
            FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION,
            FiniteState.DECREASING_JOINT3_AND_JOINT5,
            FiniteState.WAITING_DECREASING_JOINT3_AND_JOINT5,
            FiniteState.JOINT1_TURNING,
            FiniteState.WAITING_JOINT1_TURNING,
            FiniteState.LIFTING_JOINT3,
            FiniteState.WAITING_LIFTING_JOINT3,
            # FiniteState.PLACING_POSITION_TO_PRE_ORIGIN_POSITION,
            FiniteState.PLACING_POSITION_TO_ORIGIN_POSITION,
            FiniteState.RESETTING_ALL_JOINTS
        ]: 
            self.first_robot_is_carrying = True
            
        if self.second_robot_status in [
            RLRobotFiniteState.PICKING_OBJECT,
            RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION,
            RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1,
            RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2,
            RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION,
            RLRobotFiniteState.PLACING_OBJECT_UPPER,
            RLRobotFiniteState.PLACING_OBJECT_LOWER,
            RLRobotFiniteState.PLACING_OBJECT
        ]:
            self.second_robot_is_picking = True
        else :
            self.second_robot_is_picking = False
            
        if self.first_robot_is_carrying or self.second_robot_is_picking:
        
            if self.second_robot_status == RLRobotFiniteState.IDLE:
                
                left_joint_id, _, right_joint_id, _ = self._get_placed_object_info()
                if left_joint_id is not None:
                    self.robot_2_target_position_x_y = self.picking_positions[0]
                elif right_joint_id is not None:
                    self.robot_2_target_position_x_y = self.picking_positions[1]
                    
                # print(f"Robot 2 target position: {self.robot_2_target_position_x_y}")
                
                if action == 0:
                    self._brake_robot2()
                elif action == 2:
                    action_switch = self._predict_navigation_action_robot_2(self.robot_2_target_position_x_y, Direction.FORWARD, 0.2)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.PICKING_OBJECT
                elif action == 9:
                    self._back_car_robot2()
                elif action == 10:
                    self._forward_car_robot2()
            
            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION and \
                self.shared_state["current_object_index"] >= len(self.object_joint_ids):
                self.second_robot_status = RLRobotFiniteState.MOVING_TO_ORIGIN_POSITION
            
            elif self.second_robot_status == RLRobotFiniteState.MOVING_TO_ORIGIN_POSITION:
                if action == 0:
                    self._brake_robot2()
                elif action == 8:
                    action_switch = self.robot_2_navigation_to_origin_place()
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.WAIT_FOR_FINISH
                elif action == 9:
                    self._back_car_robot2()
                elif action == 10:
                    self._forward_car_robot2()
                
            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:
                
                left_joint_id, _, right_joint_id, _ = self._get_placed_object_info()
                if left_joint_id is not None:
                    self.robot_2_target_position_x_y = self.picking_positions[0]
                elif right_joint_id is not None:
                    self.robot_2_target_position_x_y = self.picking_positions[1]
                
                if action == 0:
                    self._brake_robot2()
                elif action == 2:
                    action_switch = self._predict_navigation_action_robot_2(self.robot_2_target_position_x_y, Direction.BACKWARD, 0.2)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.PICKING_OBJECT
                elif action == 9:
                    self._back_car_robot2()
                elif action == 10:
                    self._forward_car_robot2()
                
            elif self.second_robot_status == RLRobotFiniteState.PICKING_OBJECT:
                if action == 3:
                    action_switch = self._picking_object()
                if action_switch:
                    self.second_robot_status = RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION
            
            elif self.second_robot_status == RLRobotFiniteState.MAKE_DECISION_ON_PLACING_POSITION:
                if action == 0:
                    self._brake_robot2()
                elif action == 4:
                    self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1
                elif action == 5:
                    self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2
            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_1:
                
                if action == 0:
                    self._brake_robot2()
                elif action == 4:
                    self.robot_2_target_position_x_y = self.placing_positions[0]
                    action_switch = self._predict_navigation_action_robot_2(self.robot_2_target_position_x_y, Direction.FORWARD, 0.4)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT
            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION_2:
                if action == 0:
                    self._brake_robot2()
                elif action == 5:
                    self.robot_2_target_position_x_y = self.placing_positions[1]
                    action_switch = self._predict_navigation_action_robot_2(self.robot_2_target_position_x_y, Direction.FORWARD, 0.4)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT
                elif action == 9:
                    self._back_car_robot2()
                elif action == 10:
                    self._forward_car_robot2()
            
            elif self.second_robot_status == RLRobotFiniteState.PLACING_OBJECT:
                
                if action == 0:
                    self._brake_robot2()
                elif action == 6:
                    self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT_UPPER
                elif action == 7:
                    self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT_LOWER
            
            elif self.second_robot_status == RLRobotFiniteState.PLACING_OBJECT_UPPER:
                if action == 6:
                    action_switch = self._placing_object(Layer.UPPER)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION
            
            elif self.second_robot_status == RLRobotFiniteState.PLACING_OBJECT_LOWER:
                if action == 7:
                    action_switch = self._placing_object(Layer.LOWER)
                    if action_switch:
                        self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION
            
            elif self.second_robot_status == RLRobotFiniteState.WAIT_FOR_FINISH:
                if action == 0:
                    self._brake_robot2()   
        else:
            self._brake_robot2()
            
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:
            self.target_position_x_y = [-2, 1]
            navigation_obs = self._get_navigation_obs()
            action, _ = self.sec_robot_navigation_backward_model.predict(navigation_obs, deterministic=True)
            # print("moving to 'origin position'")

            robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
            dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
            reached = (dist_to_target < 0.20)
            if reached:
                action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)
                
                rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
                rover_qvel_start = self.model.jnt_dofadr[rover_joint_id]
                linear_vel = self.data.qvel[rover_qvel_start:rover_qvel_start+3]
                speed = np.linalg.norm(linear_vel)

                if speed < 0.0001: 
                    self.stop_wait_steps += 1
                    
                    if self.stop_wait_steps >= self.required_stop_steps:
                        self.second_robot_status = RLRobotFiniteState.WAIT_FOR_FINISH
                        self.break_count += 1
                else:
                    self.stop_wait_steps = 0

            self._apply_navigation_action(action)
        
        if action_switch:
           reward += 4000

        # if self.second_robot_status != RLRobotFiniteState.WAIT_FOR_FINISH:
        #     if action == 0:
        #         reward -= -0.1
        #     if action != 0:
        #         reward += 0.1
                
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        
        if distance < 0.5:
            reward -= 0.4

        return reward, action_switch

    def _brake_robot2(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + 1
        ] = [0]
        
    def _back_car_robot2(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + 2
        ] = [-3, 0]
        
    def _forward_car_robot2(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + 2
        ] = [3, 0]

    def _apply_navigation_action(self, action):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH
        ] = action

    def _predict_navigation_action_robot_2(self, target_position, direction, reach_threshold):
        action_switch = False
        navigation_obs = self._get_navigation_obs()
        
        if direction == Direction.FORWARD:
            navigation_model = self.sec_robot_navigation_forward_model
        else:
            navigation_model = self.sec_robot_navigation_backward_model
        
        action, _ = navigation_model.predict(navigation_obs, deterministic=True)

        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
        dist_to_target = np.linalg.norm(robot_2_rover_pos - target_position)
        reached = (dist_to_target < reach_threshold)
        if reached:
            action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)
            
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
            rover_qvel_start = self.model.jnt_dofadr[rover_joint_id]
            linear_vel = self.data.qvel[rover_qvel_start:rover_qvel_start+3]
            speed = np.linalg.norm(linear_vel)

            if speed < 0.0001: 
                self.stop_wait_steps += 1                
                if self.stop_wait_steps >= self.required_stop_steps:
                    action_switch = True
            else:
                self.stop_wait_steps = 0
                
        self._apply_navigation_action(action)
        
        return action_switch
    
    def _picking_object(self):
        action_switch = False
        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        if left_joint_id is not None:
            self.active_position = left_position
            self.active_joint_id = left_joint_id
            self.side = "left"
        elif right_joint_id is not None:
            self.active_position = right_position
            self.active_joint_id = right_joint_id
            self.side = "right"
            
        if self.robot_2_random_picking_steps is None:
            self.robot_2_random_picking_steps = random.randint(400, 1000)
            
        self.robot_2_random_picking_count += 1
        
        if self.robot_2_random_picking_count >= self.robot_2_random_picking_steps: 
            target_position = self.data.xpos[self.robot_2_rover_id].copy()
            # target_position[1] += 1
            target_position[2] += 0.05
            self._move_object_to_position(self.active_joint_id, target_position)
            action_switch = True
            self.robot_2_random_picking_steps = None
            self.robot_2_random_picking_count = 0
        
        return action_switch
    
    def _placing_object(self, layer):
        action_switch = False
        if self.robot_2_random_placing_steps is None:
            if layer == Layer.UPPER:
                self.robot_2_random_placing_steps = random.randint(600, 1000)
            else:
                self.robot_2_random_placing_steps = random.randint(200, 400)

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
            self._move_object_to_position(self.active_joint_id, self.robot_2_target_placing_position)
            self.robot_2_target_placing_position = None
            action_switch = True
            self.robot_2_random_placing_steps = None
            self.robot_2_random_placing_count = 0
            
        return action_switch
    
    def robot_2_navigation_to_origin_place(self):
        action_switch = False
        self.robot_2_target_position_x_y = self.robot_2_origin_place_x_y
        navigation_obs = self._get_navigation_obs()
        action, _ = self.sec_robot_navigation_backward_model.predict(navigation_obs, deterministic=True)
        # print("moving to 'origin position'")

        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
        dist_to_target = np.linalg.norm(robot_2_rover_pos - self.robot_2_target_position_x_y)
        reached = (dist_to_target < 0.20)
        if reached:
            action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)
            
            rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
            rover_qvel_start = self.model.jnt_dofadr[rover_joint_id]
            linear_vel = self.data.qvel[rover_qvel_start:rover_qvel_start+3]
            speed = np.linalg.norm(linear_vel)

            if speed < 0.0001: 
                self.stop_wait_steps += 1
                
                if self.stop_wait_steps >= self.required_stop_steps:
                    action_switch = True
            else:
                self.stop_wait_steps = 0
                
        self._apply_navigation_action(action)
        
        return action_switch
            
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
        joint_type = self.model.jnt_type[joint_id]
        
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[qpos_adr:qpos_adr+3] = new_position 
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  
            dof_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0


    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
            
    def _update_robot_tracking(self):
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        self.robot1_recent_positions.append(robot1_pos.copy())
        if len(self.robot1_recent_positions) > 3:
            self.robot1_recent_positions.pop(0)
        
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        self.robot2_recent_positions.append(robot2_pos.copy())
        if len(self.robot2_recent_positions) > 3:
            self.robot2_recent_positions.pop(0)
    
    def _check_robot_robot_collision(self):
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
                print(f"🚨 OBJECT-FLOOR COLLISION: {geom1_name} <-> {geom2_name}")
                return True
        
        return False
    
    def _check_robot_forbidden_collision(self):
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
        
        
    def _get_navigation_obs(self):
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]  
        
        try:
            robot2_body_id = self.robot_2_rover_id
            robot2_vel = self.data.cvel[robot2_body_id][:2]  
        except:
            robot2_vel = np.zeros(2)
        
        robot2_quat = self.data.xquat[self.robot_2_rover_id]
        robot2_orientation = self._quaternion_to_yaw(robot2_quat)

        robot1_predicted_trajectory = self._predict_robot1_trajectory()
        collision_risks, min_distances = self._calculate_collision_risk_timeline(robot2_pos, robot1_predicted_trajectory)
        
        trajectory_features = self._extract_trajectory_features(robot1_predicted_trajectory, robot2_pos)

        if self.robot_2_target_position_x_y is None:
            target_pos = robot2_pos.copy()  
        else:
            target_pos = np.array(self.robot_2_target_position_x_y)
        target_rel = target_pos - robot2_pos  
        target_distance = np.linalg.norm(target_rel) 
        target_angle = np.arctan2(target_rel[1], target_rel[0])  
        
        target_relative_angle = target_angle - robot2_orientation
        target_relative_angle = np.arctan2(np.sin(target_relative_angle), np.cos(target_relative_angle))  # 标准化到[-π,π]
        
        target_rel_normalized = target_rel / self.max_distance  

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])
        robot1_relative_angle = robot1_angle - robot2_orientation
        robot1_relative_angle = np.arctan2(np.sin(robot1_relative_angle), np.cos(robot1_relative_angle))
        robot1_rel_normalized = robot1_rel / self.max_distance

        walls = {"left": -3.0, "right": 3.0, "front": 3.0, "back": -3.0}
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],   
            walls["right"] - robot2_pos[0],  
            robot2_pos[1] - walls["back"],   
            walls["front"] - robot2_pos[1]   
        ])

        placing_place_1_pos = np.array([2.8, 1.0])  
        placing_1_rel = placing_place_1_pos - robot2_pos
        placing_1_distance = np.linalg.norm(placing_1_rel)
        placing_1_angle = np.arctan2(placing_1_rel[1], placing_1_rel[0]) - robot2_orientation
        placing_1_angle = np.arctan2(np.sin(placing_1_angle), np.cos(placing_1_angle))
        placing_1_rel_normalized = placing_1_rel / self.max_distance

        placing_place_2_pos = np.array([2.8, -1.0])
        placing_2_rel = placing_place_2_pos - robot2_pos
        placing_2_distance = np.linalg.norm(placing_2_rel)
        placing_2_angle = np.arctan2(placing_2_rel[1], placing_2_rel[0]) - robot2_orientation
        placing_2_angle = np.arctan2(np.sin(placing_2_angle), np.cos(placing_2_angle))
        placing_2_rel_normalized = placing_2_rel / self.max_distance

        observation = np.concatenate([
            robot2_pos / self.max_position,                    
            robot2_vel / self.max_speed,                       
            [robot2_orientation / np.pi],                 

            target_rel_normalized,                        
            [target_distance / self.max_distance],             
            [target_angle / np.pi],                       
            [target_relative_angle / np.pi],              

            robot1_rel_normalized,                        
            [robot1_distance / self.max_distance],           
            [robot1_relative_angle / np.pi],             

            trajectory_features,

            [np.max(collision_risks),                    
             np.mean(collision_risks),                   
             np.min(min_distances),                      
             np.argmax(collision_risks) / self.prediction_steps],  
            
            wall_distances / self.max_distance,                

            placing_1_rel_normalized,                     
            [placing_1_distance / self.max_distance],        
            [placing_1_angle / np.pi],                   
            
            placing_2_rel_normalized,                     
            [placing_2_distance / self.max_distance],          
            [placing_2_angle / np.pi],                    
        ], dtype=np.float32)

        return observation
    
    def _quaternion_to_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def _predict_robot1_trajectory(self):
        current_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        if len(self.robot1_recent_positions) < 2:
            return np.tile(current_pos, (self.prediction_steps, 1))
        
        velocity = self.robot1_recent_positions[-1] - self.robot1_recent_positions[-2]
        
        if len(self.robot1_recent_positions) >= 3:
            prev_velocity = self.robot1_recent_positions[-2] - self.robot1_recent_positions[-3]
            acceleration = velocity - prev_velocity
        else:
            acceleration = np.zeros(2)
        
        predicted_positions = []
        dt = 0.005 * 4
        
        for t in range(1, self.prediction_steps + 1):
            predicted_pos = (current_pos + 
                            velocity * t * dt + 
                            0.5 * acceleration * (t * dt) ** 2)
            predicted_positions.append(predicted_pos)
        
        return np.array(predicted_positions)

    def _calculate_collision_risk_timeline(self, robot2_pos, predicted_trajectory):
        robot2_future_positions = self._predict_robot2_trajectory(robot2_pos)
        
        collision_risks = []
        min_distances = []
        
        for t in range(len(predicted_trajectory)):
            if t < len(robot2_future_positions):
                distance = np.linalg.norm(predicted_trajectory[t] - robot2_future_positions[t])
            else:
                distance = np.linalg.norm(predicted_trajectory[t] - robot2_pos)
            
            min_distances.append(distance)
            
            if distance < 0.5:  
                risk = 1.0 - (distance / 0.5)
            elif distance < 1.0: 
                risk = 0.5 * (1.0 - (distance - 0.5) / 0.5)
            else:
                risk = 0.0
                
            collision_risks.append(risk)
        
        return np.array(collision_risks), np.array(min_distances)

    def _predict_robot2_trajectory(self, current_pos):
        if len(self.robot2_recent_positions) >= 2:
            velocity = self.robot2_recent_positions[-1] - self.robot2_recent_positions[-2]
        
            if len(self.robot2_recent_positions) >= 3:
                prev_velocity = self.robot2_recent_positions[-2] - self.robot2_recent_positions[-3]
                acceleration = velocity - prev_velocity
            else:
                acceleration = np.zeros(2)
                
            predicted_positions = []
            dt = 0.005 * 4
            
            for t in range(1, self.prediction_steps + 1):
                predicted_pos = (current_pos + 
                            velocity * t * dt + 
                            0.5 * acceleration * (t * dt) ** 2)
                predicted_positions.append(predicted_pos)
            
            return np.array(predicted_positions)

        return np.tile(current_pos, (self.prediction_steps, 1))

    def _calculate_collision_risk_timeline(self, robot2_pos, predicted_trajectory):
        robot2_future_positions = self._predict_robot2_trajectory(robot2_pos)
        
        collision_risks = []
        min_distances = []
        
        for t in range(len(predicted_trajectory)):
            if t < len(robot2_future_positions):
                distance = np.linalg.norm(predicted_trajectory[t] - robot2_future_positions[t])
            else:
                distance = np.linalg.norm(predicted_trajectory[t] - robot2_pos)
            
            min_distances.append(distance)
            
            if distance < 0.5: 
                risk = 1.0 - (distance / 0.5)
            elif distance < 1.0: 
                risk = 0.5 * (1.0 - (distance - 0.5) / 0.5)
            else:
                risk = 0.0
                
            collision_risks.append(risk)
        
        return np.array(collision_risks), np.array(min_distances)


    def _extract_trajectory_features(self, predicted_trajectory, robot2_pos):
        if len(predicted_trajectory) == 0:
            return np.zeros(8)
        
        relative_trajectory = predicted_trajectory - robot2_pos
        
        start_pos = relative_trajectory[0]                   
        end_pos = relative_trajectory[-1]                     
        
        trajectory_vector = end_pos - start_pos
        trajectory_length = np.linalg.norm(trajectory_vector)
        trajectory_direction = trajectory_vector / (trajectory_length + 1e-8)
        
        if len(relative_trajectory) > 2:
            vectors = np.diff(relative_trajectory, axis=0)
            direction_changes = np.diff(vectors, axis=0)
            curvature = np.mean(np.linalg.norm(direction_changes, axis=1))
        else:
            curvature = 0.0
        
        distances = np.linalg.norm(relative_trajectory, axis=1)
        min_distance = np.min(distances)
        closest_time = np.argmin(distances) / len(distances)  
        
        features = np.concatenate([
            start_pos / 8.0,                    
            trajectory_direction,               
            [trajectory_length / 8.0],          
            [curvature],                       
            [min_distance / 8.0],               
            [closest_time],                    
        ])
        
        return features
    
    def _get_object_ids(self, model):
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
    
    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        threading.Thread(
            target=place_object_on_table,
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
            model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids, remover_shared_state
        )

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        future2 = self.executor.submit(
            remove_object_on_plane_with_step_counter_with_flag,
            model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids, remover_shared_state
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
