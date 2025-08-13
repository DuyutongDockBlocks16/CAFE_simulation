import gymnasium as gym
from simplized_hybrid_controller import SimpleHybridController
import numpy as np
import mujoco

class HybridMuJoCoEnv(gym.Env):
    def __init__(self, action_repeat=40):
        super().__init__()
        self.action_repeat = action_repeat
        
        self.simple_hybrid_controller = SimpleHybridController(
            sec_robot_forward_model_path="models/final_model_continued_21600K_20250707_165449.zip.bak",
            sec_robot_backward_model_path="models/final_model_continued_56000K_20250724_140118.zip.bak",
            sec_robot_placing_model_path=None
        )
        
        self.model, self.data = self.simple_hybrid_controller.get_model_and_data()
        
        self.robot_1_rover_id = self.simple_hybrid_controller.robot_1_rover_id
        self.robot_2_rover_id = self.simple_hybrid_controller.robot_2_rover_id
        
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
            
        self.robot1_recent_positions = []
        self.robot2_recent_positions = []
        self.prediction_steps = 5
        
        obs = self._get_obs()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        # 0 for no action, 1 for brake
        self.action_space = gym.spaces.Discrete(2)
        
        self.current_step = 0
        self.max_steps = 500000
    
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
        
        if action == 1:
            self.simple_hybrid_controller.brake_robot2()
        else:
            pass
        
        break_flag = self.simple_hybrid_controller.step_for_training()
        self.current_step += 1
        
        terminated = False
        truncated = False
        obs = self._get_obs()
        reward = self._reward_function(action)
        
        if self._check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 20
            terminated = True
            
        if break_flag:
            print("Task completed successfully! Terminating episode.")
            reward += 10
            terminated = True
        
        if self.current_step >= self.max_steps:
            terminated = True
        
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _reward_function(self, action):
        reward = 0.0
        
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        
        if distance < 0.5:
            reward -= 2
        
        if action == 0:  # 如果
            reward += 1
        elif action == 1:  
            if distance < 0.5:
                reward += 3
            else:
                reward -= 1
            
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