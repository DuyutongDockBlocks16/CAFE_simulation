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

FIRST_ROBOT_ACTION_SPACE_LENGTH = 8
SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH = 2
SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH = 6

class HybridController:
    def __init__(self, 
            sec_robot_forward_model_path,
            sec_robot_backward_model_path,
            sec_robot_picking_model_paths,
            sec_robot_placing_model_path
        ):
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]

        self.model, self.data = self._get_data_and_model()

        mujoco.mj_step(self.model, self.data)
        
        self.first_robot_controller = MirobotController(self.model, self.data, self.left_object_position, self.right_object_position)
        
        object_ids = self._get_object_ids(self.model)
        
        self.object_joint_ids = []
        
        for i in object_ids:
            joint_name = f"object{i}:joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.object_joint_ids.append((i, joint_id))
        
        self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        self.robot2_rover_id = self.robot_2_rover_id

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
        for i in range(len(sec_robot_picking_model_paths)):
            self.pick_models.append(PPO.load(sec_robot_picking_model_paths[i]))
        # self.sec_robot_picking_model_0 = PPO.load(sec_robot_picking_model_paths[0])
        # self.sec_robot_picking_model_1 = PPO.load(sec_robot_picking_model_paths[1])
        self.picking_model_index = 0
        self._start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)

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
    
    def step(self):
        break_flag = False
        
        self.first_robot_status = hybrid_controller.get_first_robot_status()
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
        # if False:
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
                reached = (dist_to_target < 0.15)
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
                            self.break_count += 1
                            # if self.break_count == 3:
                            #     break_flag = True
                            # self.stop_wait_steps = 0  
                            # print("✅ Robot2已停稳，切换到PICKING_OBJECT状态")
                    else:
                        # 速度还太快，重置计数器
                        self.stop_wait_steps = 0
                        # print(f"🔄 速度过快，重置停稳计数 (速度: {speed:.4f})")
                    
                    
                    # rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
                    # rover_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot2:centroid")
                    # rover_qvel_start = self.model.jnt_dofadr[rover_joint_id]
                    # self.data.qvel[rover_qvel_start:rover_qvel_start+6] = 0.0
                    # print speed of rover
                    # print(f"Rover speed: {self.data.cvel[rover_body_id]}")

                    # self.break_count += 1
                    # if self.break_count == 2:
                    #     break_flag = True

                    
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

                if self.current_action_repeat_count == 0:
                    # 计数器为0，需要获取新的action
                    picking_obs = self._get_picking_obs()
                    action, _ = self.pick_models[self.picking_model_index].predict(picking_obs, deterministic=True)
                    self.cached_picking_action = action  # 缓存这个action
                    # print(f"🎯 获取新的picking action: {action}")
                else:
                    # 使用缓存的action
                    action = self.cached_picking_action
                    # print(f"🔄 使用缓存的action (第{self.current_action_repeat_count + 1}次): {action}")
                
                self.current_action_repeat_count += 1
                if self.current_action_repeat_count >= self.action_repeat:
                    self.current_action_repeat_count = 0  # 重置计数器
                
                picked = False

                touched = False
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "robot2:vacuum_touch")
                sensor_data = self.data.sensordata[sensor_id]
                # print(f"Vacuum position: {vacuum_pos}, Target position: {target_pos}, Distance: {distance}, Sensor data: {sensor_data}")
                if sensor_data > 0 and self._check_robot_object_collision():
                    touched = True

                suction_activated = False
                adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
                adhere_control = self.data.ctrl[adhere_actuator_id]
                if adhere_control == 1.0:
                    suction_activated = True

                # suction_activated = True
                
                picked = touched and suction_activated
                if picked:
                    print("Object picked successfully, suction activated.")
                    self.picking_stable_steps += 1
                # if picked and not self.is_object_attached:
                #     if self.active_joint_id is not None:
                #         print(self.active_joint_id, "is the active joint id")
                #         target_body_id = self.model.jnt_bodyid[self.active_joint_id]
                #         self._activate_vacuum_constraint(target_body_id)
                #     print("Object picked successfully, suction activated.")
                #     self.picking_stable_steps += 1
                # elif picked and self.is_object_attached:
                #     # 物体已固定，继续计数
                #     self.picking_stable_steps += 1
                # else:
                #     # 失去接触，但如果有约束就不重置
                #     if not self.is_object_attached:
                #         self.picking_stable_steps = 0
                    
                if self.picking_stable_steps >= 25:
                    print(self.picking_stable_steps, "steps of stable picking detected.")
                    # target_body_id = self.model.jnt_bodyid[self.active_joint_id]
                    # self._activate_vacuum_constraint(target_body_id)
                    self.picking_stable_steps = 0
                    self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION
                    action = [1.0, 0.0, 0.0, 0.05, 0.0, 0.0]  # 停止导航
                    print("Object picked successfully, moving to placing position.")
                    print(f"Switched to picking model index: {self.picking_model_index}")
                    
                # set action[0] to 0
                # action[0] = 0.0

                self._apply_picking_action(action)

            elif self.second_robot_status == RLRobotFiniteState.NAVIGATE_TO_PLACING_POSITION:
                self.data.ctrl[
                    FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
                    FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
                ] = [1.0, 0.0, 0.0, 0.05, 0.0, 0.0] 
                self.target_position_x_y = self.placing_positions[0]
                navigation_obs = self._get_navigation_obs()
                action, _ = self.sec_robot_navigation_forward_model.predict(navigation_obs, deterministic=True)
                # action, _ = self.sec_robot_navigation_backward_model.predict(navigation_obs, deterministic=True)
                
                robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
                dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
                reached = (dist_to_target < 0.35)
                if reached:
                    print("Reached placing position, preparing to place object.")
                    self.second_robot_status = RLRobotFiniteState.PLACING_OBJECT
                    self.picking_model_index = (self.picking_model_index + 1) % len(self.pick_models)
                    action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)

                    rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
                    self.data.cvel[rover_body_id] = 0.0
                    
                self._apply_navigation_action(action)
            
            elif self.second_robot_status == RLRobotFiniteState.PLACING_OBJECT:
                adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
                # self.data.ctrl[adhere_actuator_id] = 0.0  # 停止吸附
                
                # self._apply_zero_action()

                # self.second_robot_status = RLRobotFiniteState.NAVIGATE_TO_PICKING_POSITION
                break_flag = True
        else:
            self._brake_robot2()

        return break_flag

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while True:
                # sleep(0.01)
                break_flag = self.step()
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
                step += 1
                
                rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
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

    def _get_picking_obs(self):
        # 🎯 基础信息
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        robot2_quat = self.data.xquat[self.robot2_rover_id]
        robot2_orientation = self._quaternion_to_yaw(robot2_quat)
        
        # 🎯 获取目标位置
        if self.active_joint_id is not None:
            body_id = self.model.jnt_bodyid[self.active_joint_id]
            target_position = self.data.xpos[body_id]
        
        vacuum_contact_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot2:vacuum_contact_site")
        vacuum_contact_site_pos = self.data.site_xpos[vacuum_contact_site_id]
        # print(f"✅ 使用vacuum_contact_site位置: {vacuum_contact_site_pos}")

        # 🎯 计算contact_site到目标的信息
        contact_to_target_rel = target_position - vacuum_contact_site_pos
        contact_to_target_distance = np.linalg.norm(contact_to_target_rel)
        contact_to_target_angle_xy = np.arctan2(contact_to_target_rel[1], contact_to_target_rel[0])
        contact_to_target_angle_z = np.arctan2(contact_to_target_rel[2], np.linalg.norm(contact_to_target_rel[:2]))
        
        # 🎯 vacuum_sphere的速度和朝向信息
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_sphere_pos = self.data.xpos[vacuum_sphere_body_id]
        vacuum_sphere_vel = self.data.cvel[vacuum_sphere_body_id]
        vacuum_sphere_quat = self.data.xquat[vacuum_sphere_body_id]
        
        # 🎯 控制信号
        adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
        adhere_control = self.data.ctrl[adhere_actuator_id]
        
        joint1_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint1")
        joint1_control_raw = self.data.ctrl[joint1_actuator_id]
        if self.picking_model_index >= 2:
            joint1_control = (joint1_control_raw - (-10)) / (10 - (-10)) * 2 - 1
        else:
            joint1_control = (joint1_control_raw - (-1.919)) / (2.792 - (-1.919)) * 2 - 1
        
        joint2_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint2")
        joint2_control_raw = self.data.ctrl[joint2_actuator_id]
        if self.picking_model_index >= 2:
            joint2_control = (joint2_control_raw - (0)) / (1.222 - (0)) * 2 - 1
        else:
            joint2_control = (joint2_control_raw - (-0.611)) / (1.222 - (-0.611)) * 2 - 1
        
        
        joint3_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint3")
        joint3_control_raw = self.data.ctrl[joint3_actuator_id]
        joint3_control = (joint3_control_raw - (-1.565)) / (1.40 - (-1.565)) * 2 - 1
        
        joint4_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint4")
        joint4_control_raw = self.data.ctrl[joint4_actuator_id]
        joint4_control = joint4_control_raw / 3.142
        
        joint5_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint5")
        joint5_control_raw = self.data.ctrl[joint5_actuator_id]
        joint5_control = (joint5_control_raw - (-1.8)) / (2.2 - (-1.8)) * 2 - 1
        
        # 🎯 传感器数据
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "robot2:vacuum_touch")
        sensor_data = self.data.sensordata[sensor_id]
        if sensor_data > 0 and self._check_robot_object_collision():
            sensor_data = 1.0
        
        # 🎯 归一化参数
        max_position = 3.0
        max_distance = 1.0
        max_speed = 15.0
        
        # 🎯 简化的观测空间
        observation = np.concatenate([
            # 基础位置和朝向信息
            robot2_pos / max_position,                                    # [3] - 机器人位置
            [robot2_orientation / np.pi],                                 # [1] - 机器人朝向
            
            # vacuum接触点位置和速度
            vacuum_contact_site_pos / max_position,                       # [3] - contact site位置
            vacuum_sphere_vel[:3] / max_speed,                           # [3] - vacuum sphere线速度
            
            # 控制信号
            [adhere_control],                                            # [1] - 吸附控制
            [joint1_control],                                            # [1] - 关节1控制
            [joint2_control],                                            # [1] - 关节2控制
            [joint3_control],                                            # [1] - 关节3控制
            [joint4_control],                                            # [1] - 关节4控制
            [joint5_control],                                            # [1] - 关节5控制
            [sensor_data],                                               # [1] - 接触传感器
            
            # 🎯 核心：contact_site到目标的信息
            contact_to_target_rel / max_distance,                        # [3] - 相对位置
            [contact_to_target_distance / max_distance],                 # [1] - 距离
            [contact_to_target_angle_xy / np.pi],                        # [1] - 水平角度
            [contact_to_target_angle_z / np.pi],                         # [1] - 垂直角度
            
            # vacuum sphere朝向（用于对齐任务）
            vacuum_sphere_quat,                                          # [4] - 朝向四元数
            
        ], dtype=np.float32)
        
        return observation

    def _get_sphere_center_to_target_info(self, target_position):
        sphere_center, sphere_positions = self._get_vacuum_sphere_center()
        
        center_to_target_rel = target_position - sphere_center
        center_to_target_distance = np.linalg.norm(center_to_target_rel)
        
        return {
            'sphere_center': sphere_center,
            'sphere_positions': sphere_positions,
            'center_to_target_rel': center_to_target_rel,
            'center_to_target_distance': center_to_target_distance,
        }
    
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

    def _get_vacuum_sphere_center(self):
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_sphere_pos = self.data.xpos[vacuum_sphere_body_id]
        
        sphere_relative_positions = np.array([
            [0.0, 0.0, 0.0],        # 中心sphere
            [0.0027, 0.0027, 0.0],  # 右上
            [0.0027, -0.0027, 0.0], # 右下
            [-0.0027, 0.0027, 0.0], # 左上
            [-0.0027, -0.0027, 0.0], # 左下
            [-0.003, 0.0, 0.0],     # 左
            [0.003, 0.0, 0.0],      # 右
            [0.0, -0.003, 0.0],     # 下
            [0.0, 0.003, 0.0]       # 上
        ])
        
        sphere_absolute_positions = []
        
        vacuum_sphere_quat = self.data.xquat[vacuum_sphere_body_id]
        rotation_matrix = self._quaternion_to_rotation_matrix(vacuum_sphere_quat)
        
        for rel_pos in sphere_relative_positions:
            rotated_pos = rotation_matrix @ rel_pos
            absolute_pos = vacuum_sphere_pos + rotated_pos
            sphere_absolute_positions.append(absolute_pos)
        
        sphere_center = np.mean(sphere_absolute_positions, axis=0)
        
        return sphere_center, sphere_absolute_positions

    def _quaternion_to_rotation_matrix(self, quat):
        w, x, y, z = quat
        
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return rotation_matrix

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

    def _update_robot_tracking(self):
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        self.robot1_recent_positions.append(robot1_pos.copy())
        if len(self.robot1_recent_positions) > 3:
            self.robot1_recent_positions.pop(0)
        
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        self.robot2_recent_positions.append(robot2_pos.copy())
        if len(self.robot2_recent_positions) > 3:
            self.robot2_recent_positions.pop(0)


    def _apply_navigation_action(self, action):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH
        ] = action
        
    # def _apply_picking_action_with_repeat(self, action):
    #     """应用带重复的picking动作"""
        
    #     # 重复执行action_repeat次
    #     for _ in range(self.action_repeat):
    #         self._apply_picking_action(action)

    def _apply_picking_action(self, action):
        # print(f"Applying picking action: {action}")
        normalized_action = np.clip(action, -1, 1)
        if self.picking_model_index >= 2:
            real_action = self.low_bounds_for_2 + (normalized_action + 1) * (self.high_bounds - self.low_bounds_for_2) / 2
            self.data.ctrl[
                FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
                FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
            ] = real_action
        else:
            real_action = self.low_bounds_for_0_and_1 + (normalized_action + 1) * (self.high_bounds - self.low_bounds_for_0_and_1) / 2
            self.data.ctrl[
                FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
                FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
            ] = action
        
    def _apply_zero_action(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH:
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH +SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH
        ] = np.zeros(SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH, dtype=np.float32)
        
        robot2_joint_names = [
                    "robot2:Joint1",
                    "robot2:Joint2", 
                    "robot2:Joint3",
                    "robot2:Joint4",
                    "robot2:Joint5",
                    "robot2:Joint6"
                ]
                 
        for joint_name in robot2_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                
                # 🔥 使用正确的API：jnt_dofadr
                joint_qvel_start = self.model.jnt_dofadr[joint_id]
                
                # 🔥 获取关节的DOF数量（通过检查下一个关节的起始位置）
                if joint_id + 1 < self.model.njnt:
                    next_joint_qvel_start = self.model.jnt_dofadr[joint_id + 1]
                    joint_dof = next_joint_qvel_start - joint_qvel_start
                else:
                    # 最后一个关节，计算剩余的DOF
                    joint_dof = self.model.nv - joint_qvel_start
                
                # 🔥 同时获取qpos的地址
                joint_qpos_start = self.model.jnt_qposadr[joint_id]
                
                # 🔥 重置位置和速度
                self.data.qpos[joint_qpos_start:joint_qpos_start + joint_dof] = 0.0
                self.data.qvel[joint_qvel_start:joint_qvel_start + joint_dof] = 0.0
                
                # print(f"✅ 重置关节 {joint_name} (joint_id={joint_id}, dof={joint_dof})")
                
            except Exception as e:
                print(f"❌ 重置关节 {joint_name} 失败: {e}")
        
    def _brake_robot2(self):
        self.data.ctrl[
            FIRST_ROBOT_ACTION_SPACE_LENGTH :
            FIRST_ROBOT_ACTION_SPACE_LENGTH + 1
        ] = [0]
        
    def _start_object_remover_threads(self, model, data, object_joint_ids):
        # lower plane parameters
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23

        threading.Thread(
            target=remove_object_on_plane_with_step_counter,
            args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
            daemon=True
        ).start()

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        threading.Thread(
            target=remove_object_on_plane_with_step_counter,
            args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
            daemon=True
        ).start()


    def _start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        threading.Thread(
            target=place_object_on_table,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()

    def get_first_robot_status(self):
        return self.first_robot_controller.get_status()
    
    def _check_robot_object_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if ((body1_id in self.vacuum_sphere_body_ids and geom2_name in self.object_geoms) or
                (body2_id in self.vacuum_sphere_body_ids and geom1_name in self.object_geoms)):
                return True
        
        return False
    
    def _activate_vacuum_constraint(self, object_body_id):
        """激活真空约束"""
        try:
            # extract the number of object_body_id to matach vacuum_attachment id
            # str: object8 -> int: 8
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_id)
            if body_name and body_name.startswith("object"):
                # 提取数字: "object8" -> 8
                object_number = int(body_name.replace("object", ""))
                print(f"🔍 提取到物体编号: {object_number}")
            else:
                print(f"❌ 无效的body name: {body_name}")
                return

            # 查找约束ID (可能需要匹配特定的约束名称)
            constraint_name = f"vacuum_attachment_object{object_number}"  # 如果约束名称包含编号
            
            print(f"🔍 查找约束名称: {constraint_name}")
            
            if self.vacuum_constraint_id is None:
                self.vacuum_constraint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_EQUALITY, constraint_name
                )
            
            # 设置约束的第二个body为目标物体
            self.model.eq_obj2id[self.vacuum_constraint_id] = object_body_id
            
            # 激活约束
            self.data.eq_active[self.vacuum_constraint_id] = 1
            
            self.attached_object_id = object_body_id
            self.is_object_attached = True
            
            print(f"✅ 激活真空约束，物体 {object_body_id} 已固定")
            
        except Exception as e:
            print(f"❌ 激活约束失败: {e}")

    def _deactivate_vacuum_constraint(self):
        """停用真空约束"""
        try:
            if self.vacuum_constraint_id is not None:
                self.data.eq_active[self.vacuum_constraint_id] = 0
                self.attached_object_id = None
                self.is_object_attached = False
                print("🔓 停用真空约束，物体已释放")
        except Exception as e:
            print(f"❌ 停用约束失败: {e}")

if __name__ == "__main__":
    # hybrid_controller = HybridController(
    #         sec_robot_forward_model_path="models/final_model_continued_21600K_20250707_165449.zip.bak",
    #         sec_robot_backward_model_path="models/final_model_continued_56000K_20250724_140118.zip.bak",
    #         sec_robot_picking_model_paths=[
    #                 "models/final_picking_model_3000K_20250722_192051.zip.bak",
    #                 "models_bak/final_picking_model_continued_13000K_20250729_102650.zip",
    #                 "models_bak/final_picking_model_5000K_20250731_124426.zip"
    #             ],
    #         # sec_robot_picking_model_path="final_picking_model_50000K_20250728_102844.zip",
    #         sec_robot_placing_model_path=None
    #     )
    # hybrid_controller.run_simulation()


    state_file = "saved_states/robot_state_20250825_153503.pkl"
    view_saved_state(state_file)
