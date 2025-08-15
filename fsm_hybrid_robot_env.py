import gymnasium as gym
from simplized_hybrid_controller import SimpleHybridController
import numpy as np
import mujoco
from mirobot_controller import MirobotController
from config.env_config import Direction, Layer, FiniteState, RLRobotFiniteState
import concurrent.futures
from util_threads.object_placer import place_object_on_table
from util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter
import threading
from stable_baselines3 import PPO



FIRST_ROBOT_ACTION_SPACE_LENGTH = 8
SECOND_ROBOT_NAVIGATION_ACTION_SPACE_LENGTH = 2
SECOND_ROBOT_PICKING_AND_PLACING_ACTION_SPACE_LENGTH = 6

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

        mujoco.mj_step(self.model, self.data)
        
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
        
        self.first_robot_status = None
        
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
        
        self.robot2_recent_positions = []
        
        self.second_robot_status = RLRobotFiniteState.IDLE

        self.robot_2_target_position_x_y = None

        # Robot2 setup end
        
        # General setup for RL robots
        self.prediction_steps = 5
        self.required_stop_steps = 50
        
        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
        ]
        
        self.max_position = 3.0     
        self.max_speed = 2.0        
        self.max_distance = 8.0
        self.active_joint_id = None
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
        # 0 Brake and wait
        # 1 Keep moving
        # 2 Move to pickingplace:table0
        # 3 Move to pickingplace:table1
        # 4 Pick
        # 5 Move to placingplace:table0
        # 6 Move to placingplace:table1
        # 7 Place Upper
        # 8 Place Lower
        self.action_space = gym.spaces.Discrete(9)

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
        self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)
        
            
        obs = self._get_obs()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.simple_hybrid_controller.reset()

        return self._get_obs(), {}
    
    def _get_obs(self):
        max_position = 3.0     
        max_speed = 3.0        
        max_distance = 8.0
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

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])
        robot1_relative_angle = robot1_angle - robot2_orientation
        robot1_relative_angle = np.arctan2(np.sin(robot1_relative_angle), np.cos(robot1_relative_angle))
        robot1_rel_normalized = robot1_rel / max_distance
        
        observation = np.concatenate([
            # 🎯 机器人自身状态
            robot2_pos / max_position,                    # [2] 绝对位置
            robot2_vel / max_speed,                       # [2] 速度 
            [robot2_orientation / np.pi],                 # [1] 朝向 

            # 🎯 机器人1相对信息（增强版）
            robot1_rel_normalized,                        # [2] 机器人1相对位置 
            [robot1_distance / max_distance],             # [1] 机器人1距离
            [robot1_relative_angle / np.pi],              # [1] 机器人1相对角度 

            trajectory_features,

            [np.max(collision_risks),                     # 最大风险
             np.mean(collision_risks),                    # 平均风险
             np.min(min_distances),                       # 最小距离
             np.argmax(collision_risks) / self.prediction_steps],  # 最危险时刻(归一化)
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
            obs, reward, terminated, truncated, info = self._original_step(action)
            total_reward += reward
            final_obs = obs
            final_info.update(info)
            
            if terminated or truncated:
                break
        
        return final_obs, total_reward, terminated, truncated, final_info
        
    def _original_step(self, action):
        self.model, self.data = self.simple_hybrid_controller.get_model_and_data()
        # print(f"Step {self.current_step + 1}/{self.max_steps}, Action: {action}")
        
        if action == 1:
            break_flag, action_switch = self.simple_hybrid_controller.step_for_training(brake_flag=True)
        else:
            break_flag, action_switch = self.simple_hybrid_controller.step_for_training()

        self.current_step += 1
        
        terminated = False
        truncated = False
        obs = self._get_obs()
        reward = self._reward_function(action, action_switch)
        
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
            reward += 1000
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
        
        return obs, reward, terminated, truncated, info
    
    def _reward_function(self, action, action_switch):
        reward = 0.0
        
        time_penalty = 0.1
        reward -= time_penalty
        
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        
        if distance < 0.5:
            reward -= 0.4
        
        # if action == 0:  # 如果
        #     reward += 1
        # elif action == 1:  
        #     if distance < 0.5:
        #         reward += 3
        #     else:
        #         reward -= 1
        
        if action_switch:
            reward += 1000
        
        return reward

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