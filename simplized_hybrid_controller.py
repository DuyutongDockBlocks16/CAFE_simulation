import gymnasium as gym
from stable_baselines3 import PPO
from time import sleep
import mujoco.viewer
import time
import os
from datetime import datetime
from config.env_config import Direction, Layer, FiniteState, RLRobotFiniteState
from utils.mujoco_state_saver import save_mujoco_state_to_file
from utils.mujoco_state_loader import load_mujoco_state_from_file, restore_mujoco_state, view_saved_state
from mirobot_controller import MirobotController
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
from util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter
import threading
import numpy as np
import pickle
import json
import random
import concurrent.futures

FIRST_ROBOT_ACTION_SPACE_LENGTH = 8
SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH = 2
SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH = 6

class SimpleHybridController:
    def __init__(self, 
            sec_robot_forward_model_path,
            sec_robot_backward_model_path,
            # sec_robot_picking_model_paths,
            sec_robot_placing_model_path
        ):
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]

        self.model, self.data = self._get_data_and_model()
        
        # collect initial model and data
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

        mujoco.mj_step(self.model, self.data)
        
        self.first_robot_controller = MirobotController(self.model, self.data, self.left_object_position, self.right_object_position)
        
        object_ids = self._get_object_ids(self.model)
        
        self.object_joint_ids = []
        
        for i in object_ids:
            joint_name = f"object{i}:joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.object_joint_ids.append((i, joint_id))
        
        self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": True}

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")

        self.placingplace1_low_plane_pos = [2.8, 1, 0.23]
        self.placingplace2_low_plane_pos = [2.8, -1, 0.23]
        self.placingplace1_high_plane_pos = [2.8, 1, 0.43]
        self.placingplace2_high_plane_pos = [2.8, -1, 0.43]

        self.picking_positions = [
            [1, -2.45], # left
            [-1, -2.45],  # right
        ]

        self.placing_positions = [
            [2.80, -1],
            [2.80, 1]
        ]

        self.robot2_arm_bodies = [
            "robot2:base",          
            "robot2:base_link",     
            "robot2:link1",         
            "robot2:link2",         
            "robot2:link3",         
            "robot2:link4",         
            "robot2:link5",         
            "robot2:link6",         
            "robot2:vacuum_sphere"
        ]

        self.robot2_body_ids = []
        for body_name in self.robot2_arm_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot2_body_ids.append(body_id)
            except:
                continue
            
        self.vacuum_sphere_body = ["robot2:vacuum_sphere"]

        self.vacuum_sphere_body_ids = []
        for body_name in self.vacuum_sphere_body:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.vacuum_sphere_body_ids.append(body_id)
            except:
                continue

        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "pickingplace:table0",
            "pickingplace:table2"
        ]
        
        self.object_geoms = [
            "object0_geom", "object1_geom", "object2_geom", "object3_geom",
            "object4_geom", "object5_geom", "object6_geom", "object7_geom",
            "object8_geom", "object9_geom"
        ]

        self.robot_arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.robot2_arm_bodies]

        self.target_position_x_y = None

        self.robot1_recent_positions = []
        self.robot2_recent_positions = []
        self.prediction_steps = 5

        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)
        
        self.max_position = 3.0     
        self.max_speed = 2.0        
        self.max_distance = 8.0
        
        self.sec_robot_navigation_forward_model = PPO.load(sec_robot_forward_model_path)
        self.sec_robot_navigation_backward_model = PPO.load(sec_robot_backward_model_path)
        self.pick_models = []
        # for i in range(len(sec_robot_picking_model_paths)):
        #     self.pick_models.append(PPO.load(sec_robot_picking_model_paths[i]))
        # self.sec_robot_picking_model_0 = PPO.load(sec_robot_picking_model_paths[0])
        # self.sec_robot_picking_model_1 = PPO.load(sec_robot_picking_model_paths[1])
        self.picking_model_index = 0
        # self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.remover_futures = []
        # self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)

        self.first_robot_status = None
        self.second_robot_status = RLRobotFiniteState.IDLE
        self.first_robot_is_carrying = False
        self.second_robot_is_picking = False

        self.active_joint_id = None

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

        self.suction_height_threshold = 0.01 
        self.suction_radius_threshold = 0.015 

        self.picking_stable_steps = 0
        self.break_count = 0
        
        self.stop_wait_steps = 0
        self.required_stop_steps = 50
        
        self.low_bounds_for_0_and_1 = np.array([-1.0, -1.919, -0.611, -1.565, -3.142, -0.2], dtype=np.float32)
        self.low_bounds_for_2 = np.array([-1.0, -10, 0, -1.565, -3.142, -0.2], dtype=np.float32)
        self.high_bounds = np.array([1.0, 10, 1.222, 1.40, 3.142, 0.2], dtype=np.float32)

        self.action_repeat = 4
        self.current_action_repeat_count = 0
        self.cached_picking_action = None
        
        self.vacuum_constraint_id = None
        self.attached_object_id = None
        self.is_object_attached = False
        
        self.robot_2_random_picking_steps = None
        self.robot_2_random_picking_count = 0
        
        self.robot_2_random_placing_steps = None
        self.robot_2_random_placing_count = 0
        
        self.simulation_step = 0
        
    def _get_data_and_model(self):
        model = mujoco.MjModel.from_xml_path("xml/scene_mirobot.xml")
        data = mujoco.MjData(model)
        time_step = 0.005
        model.opt.timestep = time_step  
        return model, data

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
    
    def step(self, brake_flag):
        break_flag = False
        action_switch = False
        
        self.first_robot_status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and self.first_robot_status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            break_flag = True

        self.first_robot_controller.step(self.shared_state["current_object_position"])

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
            RLRobotFiniteState.PLACING_OBJECT
        ]:
            self.second_robot_is_picking = True
        else :
            self.second_robot_is_picking = False

        if self.first_robot_is_carrying or self.second_robot_is_picking:
        # if self.first_robot_is_carrying:
            if self.second_robot_status == RLRobotFiniteState.IDLE:
                
                left_joint_id, _, right_joint_id, _ = self._get_placed_object_info()
                if left_joint_id is not None:
                    self.target_position_x_y = self.picking_positions[0]
                elif right_joint_id is not None:
                    self.target_position_x_y = self.picking_positions[1]

                navigation_obs = self._get_navigation_obs()
                action, _ = self.sec_robot_navigation_forward_model.predict(navigation_obs, deterministic=True)

                robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
                dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
                reached = (dist_to_target < 0.15)
                if reached:
                    self.second_robot_status = RLRobotFiniteState.PICKING_OBJECT
                    action_switch = True
                    action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)
                    self.break_count += 1
                    
                self._apply_navigation_action(action)
            
            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION:
                
                left_joint_id, _, right_joint_id, _ = self._get_placed_object_info()
                if left_joint_id is not None:
                    self.target_position_x_y = self.picking_positions[0]
                elif right_joint_id is not None:
                    self.target_position_x_y = self.picking_positions[1]

                navigation_obs = self._get_navigation_obs()
                action, _ = self.sec_robot_navigation_backward_model.predict(navigation_obs, deterministic=True)

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
                        # print(f"⏳ 停稳检测: {self.stop_wait_steps}/{self.required_stop_steps} (速度: {speed:.4f})")
                        
                        if self.stop_wait_steps >= self.required_stop_steps:
                            self.second_robot_status = RLRobotFiniteState.PICKING_OBJECT
                            action_switch = True
                            self.break_count += 1
                    else:
                        # 速度还太快，重置计数器
                        self.stop_wait_steps = 0
                        # print(f"🔄 速度过快，重置停稳计数 (速度: {speed:.4f})")
  
                self._apply_navigation_action(action)
                
            elif self.second_robot_status == RLRobotFiniteState.PICKING_OBJECT:
                
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
                    self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION
                    action_switch = True
                    self.robot_2_random_picking_steps = None
                    self.robot_2_random_picking_count = 0

            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION:
                # self.data.ctrl[
                #     FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
                #     FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
                # ] = [1.0, 0.0, 0.0, 0.05, 0.0, 0.0] 
                self.target_position_x_y = self.placing_positions[0]
                navigation_obs = self._get_navigation_obs()
                action, _ = self.sec_robot_navigation_forward_model.predict(navigation_obs, deterministic=True)
                # action, _ = self.sec_robot_navigation_backward_model.predict(navigation_obs, deterministic=True)
                
                robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
                dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
                reached = (dist_to_target < 0.40)
                if reached:
                    print("Reached placing position, preparing to place object.")
                    self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT
                    action_switch = True
                    # self.picking_model_index = (self.picking_model_index + 1) % len(self.pick_models)
                    action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)

                    rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
                    self.data.cvel[rover_body_id] = 0.0
                    
                self._apply_navigation_action(action)
            
            elif self.second_robot_status == RLRobotFiniteState.PLACING_OBJECT:
                # adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
                # self.data.ctrl[adhere_actuator_id] = 0.0  # 停止吸附
                
                # self._apply_zero_action()
                
                # # reset all robot2 joints to zero
                # for joint_id in self.robot_arm_ids:
                #     self.data.qpos[joint_id] = 0.0
                #     self.data.qvel[joint_id] = 0.0
                
                if self.robot_2_random_placing_steps is None:
                    self.robot_2_random_placing_steps = random.randint(200, 1000)

                self.robot_2_random_placing_count += 1

                if self.robot_2_random_placing_count >= self.robot_2_random_placing_steps:
                    target_position = self.placingplace2_low_plane_pos.copy()
                    # print(f"Placing object at {target_position}")
                    target_position[2] += 0.02
                    self._move_object_to_position(self.active_joint_id, target_position)
                    self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION
                    action_switch = True
                    self.robot_2_random_placing_steps = None
                    self.robot_2_random_placing_count = 0

                # break_flag = True
            elif self.second_robot_status == RLRobotFiniteState.WAIT_FOR_FINISH:
                self.brake_robot2()
                
        else:
            self.brake_robot2()
            
        if brake_flag:
            self.brake_robot2()
            
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

        return break_flag, action_switch
    
    def step_for_training(self, brake_flag=False):
        break_flag, action_switch = self.step(brake_flag=brake_flag)
        mujoco.mj_step(self.model, self.data)
        return break_flag, action_switch

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # step = 0
            while True:
                # sleep(0.01)
                break_flag, _ = self.step()
                # print position of rover
                # if break_flag:
                #     rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
                #     print(f"in Rover position: {self.data.xpos[rover_body_id]}")
                #     # [ 1.07028213 -2.31816306  0.19964572]
                if break_flag:
                    save_mujoco_state_to_file(self.model, self.data)
                    print("Simulation ended, exiting loop.")
                    break

                mujoco.mj_step(self.model, self.data)
                self.simulation_step += 1
                
                # if self.simulation_step == 40000:
                #     self.reset()
                #     self.simulation_step = 0

                # rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
                # print(f"out Rover position: {self.data.xpos[rover_body_id]}")

                # if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
                #     print("QACC error detected! Simulation unstable, exiting loop.")
                #     break
                    
                viewer.sync()

            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


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

        target_pos = np.array(self.target_position_x_y) 
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

    def _apply_navigation_action(self, action):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH
        ] = action
        
    def _apply_zero_action(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
        ] = np.zeros(SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH, dtype=np.float32)
        
    def brake_robot2(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + 1
        ] = [0]
        
    def _start_object_remover_threads(self, model, data, object_joint_ids):
        
        self._cancel_remover_tasks()
    
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
            remove_object_on_plane_with_step_counter,
            model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids
        )
        self.remover_futures.append(future1)

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        future2 = self.executor.submit(
            remove_object_on_plane_with_step_counter,
            model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids
        )
        self.remover_futures.append(future2)
        
    def _cancel_remover_tasks(self):
        for i, future in enumerate(self.remover_futures):
            try:
                cancelled = future.cancel()
                print(f"   任务 {i}: 取消{'成功' if cancelled else '失败'} (状态: {future.done()})")
            except Exception as e:
                print(f"   任务 {i}: 取消异常 {e}")

    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        threading.Thread(
            target=place_object_on_table,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()

    def get_first_robot_status(self):
        return self.first_robot_controller.get_status()
    
    def _move_object_to_position(self, joint_id, new_position):
        qpos_adr = self.model.jnt_qposadr[joint_id]
        joint_type = self.model.jnt_type[joint_id]
        
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[qpos_adr:qpos_adr+3] = new_position 
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  
            dof_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0
            
    def get_model_and_data(self):
        return self.model, self.data
    
    def reset(self):
        # self._cancel_remover_tasks()
        
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        
        if self.shared_state["stop"] is False:
            self.shared_state["stop"] = True
            self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": False}
        # self.shared_state = {"current_object_index": 0, "current_object_position": None, "stop": False, "stopped": False}
        # self.shared_state["current_object_index"] = 0

        self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)
        
        # while self.shared_state["current_object_position"] is None:
        #     print(self.shared_state["current_object_position"])
        #     sleep(0.01)   
        
        self.first_robot_status = None
        self.first_robot_controller.reset_all_joints()
        self.first_robot_controller.set_state(FiniteState.IDLE)
        
        self.second_robot_status = RLRobotFiniteState.IDLE
        
        mujoco.mj_forward(self.model, self.data)
        
    def __del__(self):
        self._cancel_remover_tasks()
        self.executor.shutdown(wait=True)
    
if __name__ == "__main__":
    simple_hybrid_controller = SimpleHybridController(
            sec_robot_forward_model_path="models/final_model_continued_21600K_20250707_165449.zip.bak",
            sec_robot_backward_model_path="models/final_model_continued_56000K_20250724_140118.zip.bak",
            sec_robot_placing_model_path=None
        )

    simple_hybrid_controller.run_simulation()
    
    
