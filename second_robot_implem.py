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
            sec_robot_navigation_model_path,
            sec_robot_picking_model_path,
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

        self.picking_positions = [
            [1, -2.45], # left
            [-1, -2.45],  # right
        ]

        self.placing_positions = [
        ]  

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

        self.sec_robot_navigation_model = PPO.load(sec_robot_navigation_model_path)
        self._start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)

        self.first_robot_status = None
        self.second_robot_status = RLRobotFiniteState.IDLE
        self.second_robot_is_active = False

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

        self.second_robot_is_active = False

        if self.first_robot_status not in [
            FiniteState.IDLE,
            FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION,
            FiniteState.DECREASING_JOINT3_AND_JOINT5,
            FiniteState.WAITING_DECREASING_JOINT3_AND_JOINT5,
            FiniteState.JOINT1_TURNING,
            FiniteState.WAITING_JOINT1_TURNING,
            FiniteState.LIFTING_JOINT3,
            FiniteState.WAITING_LIFTING_JOINT3,
            FiniteState.PLACING_POSITION_TO_ORIGIN_POSITION
        ]: 
            self.second_robot_is_active = True

        if self.second_robot_is_active:
            if self.second_robot_status == RLRobotFiniteState.IDLE:
                self.target_position_x_y = self.picking_positions[1]
                navigation_obs = self._get_navigation_obs()
                action, _ = self.sec_robot_navigation_model.predict(navigation_obs, deterministic=True)

                robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
                dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
                reached = (dist_to_target < 0.15)
                if reached:
                    self.second_robot_status = RLRobotFiniteState.PICKING_OBJECT
                    action = np.zeros(SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH, dtype=np.float32)
                    break_flag = True
                    
                self._apply_navigation_action(action)
            
        return break_flag

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while True:
                break_flag = self.step()
                if break_flag:
                    save_mujoco_state_to_file(self.model, self.data)
                    print("Simulation ended, exiting loop.")
                    break

                mujoco.mj_step(self.model, self.data)
                step += 1

                if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
                    print("QACC error detected! Simulation unstable, exiting loop.")
                    break
                    
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
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH:
            FIRST_ROBOT_ACTION_SPACE_LENGTH + SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH + SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH
        ] = action
    
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

if __name__ == "__main__":
    # hybrid_controller = HybridController(
    #         sec_robot_navigation_model_path="final_model_continued_21600K_20250707_165449.zip",
    #         sec_robot_picking_model_path=None,
    #         sec_robot_placing_model_path=None
    #     )
    # hybrid_controller.run_simulation()


    state_file = "saved_states/robot_state_20250710_162826.pkl"
    view_saved_state(state_file)



# def approach_model_implementation(env):
#     ppo_model = PPO.load(APPROACHING_MODEL_NAME, env=env)
#     print(f"✅ Loaded PPO model: {APPROACHING_MODEL_NAME}")
    
#     obs, info = env.reset()
#     print("🎬 Starting model demonstration...")
    
#     env.render()
#     sleep(15) 
    
#     episode_count = 0
#     max_episodes = 1 
    
#     while episode_count < max_episodes:
#         print(f"\n🎯 Episode {episode_count + 1}/{max_episodes}")
        
#         step_count = 0
#         max_steps_per_episode = 10000  
        
#         while step_count < max_steps_per_episode:
#             env.render()

#             action, _ = ppo_model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
            
#             step_count += 1
            
#             if step_count % 500 == 0:
#                 print(f"   Step {step_count}, Reward: {reward:.2f}")
            
#             if terminated or truncated:
#                 print(f"   Episode ended after {step_count} steps")
#                 print(f"   Final reward: {reward:.2f}")
#                 print(f"   Terminated: {terminated}, Truncated: {truncated}")
#                 # env.unwrapped.data.ctrl[:] = 0
#                 # mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)
#                 save_mujoco_state_to_file(env.unwrapped.model, env.unwrapped.data)

#                 # obs, info = env.reset()
#                 break
        
#         episode_count += 1
        
#         if episode_count < max_episodes:
#             print("   🔄 Starting next episode in 3 seconds...")
#             sleep(3)
    
#     print("🎬 Demonstration completed!")
    
#     mujoco_model = env.unwrapped.model
#     mujoco_data = env.unwrapped.data
    
#     if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
#         env.unwrapped.viewer.close()
#     env.close()
    
#     print("🔍 Launching passive viewer...")
#     sleep(5)
    
#     with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
#         print("🎮 Passive viewer launched!")
#         print("   - Press ESC to exit viewer")
#         print("   - Use mouse to rotate view")
#         print("   - Use scroll to zoom")
        
#         last_time = time.time()
#         frame_count = 0
        
#         while viewer.is_running():
#             mujoco.mj_step(mujoco_model, mujoco_data)
#             viewer.sync()
            
#             frame_count += 1
#             current_time = time.time()
            
#             if current_time - last_time >= 5.0: 
#                 fps = frame_count / (current_time - last_time)
#                 print(f"📊 Simulation FPS: {fps:.1f}")
#                 frame_count = 0
#                 last_time = current_time
    
#     print("✅ Viewer closed. Implementation demo finished!")

# def setup_initial_state(model, data):
#     mujoco.mj_resetData(model, data)
#     mujoco.mj_forward(model, data)
