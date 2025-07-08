import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController 
import threading
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
from util_threads.object_remover_step_counter import remove_object_on_plane_with_step_counter
from config.env_config import FiniteState
import random

ACTION_SPACE_REDUCTION = 14  # Number of actuators to be reduced from the action space for moving

class SecondRobotMuJoCoEnv(gym.Env):
    def __init__(self, xml_path, action_repeat=4):
        super().__init__()

        self.action_repeat = action_repeat
        if self.action_repeat > 1:
            print(f"🔄 Action Repeat enabled: {self.action_repeat}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        time_step = 0.005
        self.model.opt.timestep = time_step

        object_ids = self._get_object_ids(self.model)
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

        self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        
        # self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "placingplace1:low_plane")
        # self.target_position_x_y = [-1, -1.7] 
        self.target_positions = [
                # [2, 2], 
                # [2, 0],
                # [2, -2],
                # [2, 1],
                # [2, -1],
                # [0, 2],
                # [0, 1],
                [-1, -2.45],
                # [1, -2.45],
            ]
        
        self.picking_positions = [
                # [1, -2.45],
                [-1, -2.45],
        ]

        self.target_position_x_y = random.choice(self.picking_positions)

        self.robot1_recent_positions = []
        self.robot2_recent_positions = []
        self.prediction_steps = 5

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

        self.max_steps = 5000
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
        
        self.robot1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        self.safe_distance = 0.8
        self.prev_dist = None
        self.init_dist = None
        self.static_counter = 0
        self.max_static_steps = 400 
        self.finished = True
        self.robot2_joint_indices = self._get_robot2_joint_indices()
        self.robot2_initial_qpos = {}
        self.robot2_initial_qvel = {}
        self.robot2_initial_ctrl = {}
        self._store_robot2_initial_states()

    def reset_robot2_only(self):
        """只重置robot2的状态"""
        
        print("🔄 Resetting robot2 states only...")
        
        # 🎯 重置robot2的qpos
        reset_count = 0
        for idx, initial_value in self.robot2_initial_qpos.items():
            if idx < len(self.data.qpos):
                self.data.qpos[idx] = initial_value
                reset_count += 1
            else:
                print(f"  ⚠️ qpos index {idx} out of range during reset")
        print(f"   qpos: {reset_count} values reset")
        
        # 🎯 重置robot2的qvel
        reset_count = 0
        for idx, initial_value in self.robot2_initial_qvel.items():
            if idx < len(self.data.qvel):
                self.data.qvel[idx] = initial_value
                reset_count += 1
            else:
                print(f"  ⚠️ qvel index {idx} out of range during reset")
        print(f"   qvel: {reset_count} values reset")
        
        # 🎯 重置robot2的ctrl
        reset_count = 0
        for idx, initial_value in self.robot2_initial_ctrl.items():
            if idx < len(self.data.ctrl):
                self.data.ctrl[idx] = initial_value
                reset_count += 1
            else:
                print(f"  ⚠️ ctrl index {idx} out of range during reset")
        print(f"   ctrl: {reset_count} values reset")
        
        # 🎯 确保状态同步
        mujoco.mj_forward(self.model, self.data)
        
        print("✅ Robot2 reset complete")

    def reset(self, seed=None, options=None):
        self.target_position_x_y = random.choice(self.target_positions)
        self.robot1_recent_positions = []
        self.robot2_recent_positions = []  

        if self.finished:     
            if self.shared_state["stop"] is False:
                self.shared_state["stop"] = True
                # while self.shared_state["stopped"] is False:
                #     pass
                self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": False}
 
            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = self.initial_qvel
            self.data.ctrl[:] = self.initial_ctrl

            self._start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
            self._start_object_remover_threads(self.model, self.data, self.object_joint_ids)
            self.first_robot_controller.set_state(FiniteState.IDLE)
            self.first_robot_controller.reset_all_joints()

            self.current_world_step = 0

            mujoco.mj_forward(self.model, self.data)
            self.current_world_step = 0
            self.finished = False

            inactive_status = [
                FiniteState.IDLE,
                FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION,
                FiniteState.DECREASING_JOINT3_AND_JOINT5,
                FiniteState.WAITING_DECREASING_JOINT3_AND_JOINT5,
                FiniteState.JOINT1_TURNING,
                FiniteState.WAITING_JOINT1_TURNING,
                FiniteState.LIFTING_JOINT3,
                FiniteState.WAITING_LIFTING_JOINT3
            ]
            while self.first_robot_controller.get_status() in inactive_status:
                mujoco.mj_step(self.model, self.data)
                self.first_robot_controller.step(self.shared_state["current_object_position"])
                mujoco.mj_forward(self.model, self.data)

        if not self.finished:
            self.reset_robot2_only()

        self.current_step = 0

        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id]
        self.prev_dist = np.linalg.norm(robot_2_rover_pos[0:2] - self.target_position_x_y)
        self.init_dist = self.prev_dist

        self.prev_position = None
        self.static_counter = 0

        return self._get_obs(), {}

    def step(self, action):
        self._update_robot_tracking()
        """带有action repeat的step方法"""
        if self.action_repeat == 1:
            # 如果action_repeat=1，使用原始step逻辑
            return self._original_step(action)
        else:
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
        normalized_action = np.clip(action, -1, 1)
        real_action = self.low_bounds + (normalized_action + 1) * (self.high_bounds - self.low_bounds) / 2

        terminated = False
        truncated = False
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            truncated = True
            self.finished = True

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]  # Get only the first two coordinates (x, y)

        reward, reached = self._reward_function(robot_2_rover_pos)

        if self._check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 200
            terminated = True

        if self._check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 250
            terminated = True

        if reached:
            print("Robot2 has reached the target area! Terminating episode.")
            terminated = True

        self.current_step += 1
        self.current_world_step += 1

        if self.current_step >= self.max_steps:
            terminated = True
        
        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("QACC error detected! Simulation unstable, exiting loop.")
            truncated = True
            self.finished = True
        
        if self.current_world_step >= 500000:
            print("Maximum world steps reached! Terminating episode.")
            truncated = True
            self.finished = True
            
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        max_position = 3.0     
        max_speed = 2.0        
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

        target_pos = np.array(self.target_position_x_y) 
        target_rel = target_pos - robot2_pos  
        target_distance = np.linalg.norm(target_rel) 
        target_angle = np.arctan2(target_rel[1], target_rel[0])  
        
        target_relative_angle = target_angle - robot2_orientation
        target_relative_angle = np.arctan2(np.sin(target_relative_angle), np.cos(target_relative_angle))  # 标准化到[-π,π]
        
        target_rel_normalized = target_rel / max_distance  

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])
        robot1_relative_angle = robot1_angle - robot2_orientation
        robot1_relative_angle = np.arctan2(np.sin(robot1_relative_angle), np.cos(robot1_relative_angle))
        robot1_rel_normalized = robot1_rel / max_distance

        # 🎯 墙壁相对位置
        walls = {"left": -3.0, "right": 3.0, "front": 3.0, "back": -3.0}
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],   
            walls["right"] - robot2_pos[0],  
            robot2_pos[1] - walls["back"],   
            walls["front"] - robot2_pos[1]   
        ])

        # 🎯 放置点相对位置
        placing_place_1_pos = np.array([2.8, 1.0])  
        placing_1_rel = placing_place_1_pos - robot2_pos
        placing_1_distance = np.linalg.norm(placing_1_rel)
        placing_1_angle = np.arctan2(placing_1_rel[1], placing_1_rel[0]) - robot2_orientation
        placing_1_angle = np.arctan2(np.sin(placing_1_angle), np.cos(placing_1_angle))
        placing_1_rel_normalized = placing_1_rel / max_distance

        placing_place_2_pos = np.array([2.8, -1.0])
        placing_2_rel = placing_place_2_pos - robot2_pos
        placing_2_distance = np.linalg.norm(placing_2_rel)
        placing_2_angle = np.arctan2(placing_2_rel[1], placing_2_rel[0]) - robot2_orientation
        placing_2_angle = np.arctan2(np.sin(placing_2_angle), np.cos(placing_2_angle))
        placing_2_rel_normalized = placing_2_rel / max_distance
        
        observation = np.concatenate([
            # 🎯 机器人自身状态
            robot2_pos / max_position,                    # [2] 绝对位置
            robot2_vel / max_speed,                       # [2] 速度 
            [robot2_orientation / np.pi],                 # [1] 朝向 

            # 🎯 目标相对信息（增强版）
            target_rel_normalized,                        # [2] 目标相对位置向量 
            [target_distance / max_distance],             # [1] 目标距离
            [target_angle / np.pi],                       # [1] 目标世界角度
            [target_relative_angle / np.pi],              # [1] 目标相对角度 

            # 🎯 机器人1相对信息（增强版）
            robot1_rel_normalized,                        # [2] 机器人1相对位置 
            [robot1_distance / max_distance],             # [1] 机器人1距离
            [robot1_relative_angle / np.pi],              # [1] 机器人1相对角度 

            trajectory_features,

            [np.max(collision_risks),                     # 最大风险
             np.mean(collision_risks),                    # 平均风险
             np.min(min_distances),                       # 最小距离
             np.argmax(collision_risks) / self.prediction_steps],  # 最危险时刻(归一化)
            
            # 🎯 环境相对信息
            wall_distances / max_distance,                # [4] 墙壁距离

            # 🎯 放置点相对信息（增强版）
            placing_1_rel_normalized,                     # [2] 放置点1相对位置 
            [placing_1_distance / max_distance],          # [1] 放置点1距离
            [placing_1_angle / np.pi],                    # [1] 放置点1相对角度 
            
            placing_2_rel_normalized,                     # [2] 放置点2相对位置 
            [placing_2_distance / max_distance],          # [1] 放置点2距离
            [placing_2_angle / np.pi],                    # [1] 放置点2相对角度 
        ], dtype=np.float32)

        return observation

    def _reward_function(self, robot_2_rover_pos):

        robot2_quat = self.data.xquat[self.robot_2_rover_id]

        dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
     
        progress_reward = 0
        if self.prev_dist is not None:
            progress_amount = (self.prev_dist - dist_to_target) * 2000

            coefficient = 1.0
            
            progress = progress_amount * coefficient
            
            progress_reward = progress


        time_penalty = -0.3

        arrival_bonus = 0

        reached = (dist_to_target < 0.15)
        if reached:
            arrival_bonus = 2000

        total_reward = progress_reward + time_penalty + arrival_bonus

        self.prev_dist = dist_to_target

        return total_reward, reached

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
            # 没有历史数据 -> 静止预测
            return np.tile(current_pos, (self.prediction_steps, 1))
        
        # 🎯 计算速度和加速度
        velocity = self.robot1_recent_positions[-1] - self.robot1_recent_positions[-2]
        
        if len(self.robot1_recent_positions) >= 3:
            prev_velocity = self.robot1_recent_positions[-2] - self.robot1_recent_positions[-3]
            acceleration = velocity - prev_velocity
        else:
            acceleration = np.zeros(2)
        
        # 🎯 运动学预测（适用于所有情况）
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
            
            if distance < 0.5:  # 危险距离
                risk = 1.0 - (distance / 0.5)
            elif distance < 1.0:  # 警告距离
                risk = 0.5 * (1.0 - (distance - 0.5) / 0.5)
            else:
                risk = 0.0
                
            collision_risks.append(risk)
        
        return np.array(collision_risks), np.array(min_distances)


    def _extract_trajectory_features(self, predicted_trajectory, robot2_pos):
        if len(predicted_trajectory) == 0:
            return np.zeros(8)
        
        # 相对于Robot2的轨迹
        relative_trajectory = predicted_trajectory - robot2_pos
        
        # 特征提取
        start_pos = relative_trajectory[0]                    # 起始相对位置
        end_pos = relative_trajectory[-1]                     # 结束相对位置
        
        # 轨迹方向和长度
        trajectory_vector = end_pos - start_pos
        trajectory_length = np.linalg.norm(trajectory_vector)
        trajectory_direction = trajectory_vector / (trajectory_length + 1e-8)
        
        # 轨迹曲率（方向变化）
        if len(relative_trajectory) > 2:
            vectors = np.diff(relative_trajectory, axis=0)
            direction_changes = np.diff(vectors, axis=0)
            curvature = np.mean(np.linalg.norm(direction_changes, axis=1))
        else:
            curvature = 0.0
        
        # 最近接近点
        distances = np.linalg.norm(relative_trajectory, axis=1)
        min_distance = np.min(distances)
        closest_time = np.argmin(distances) / len(distances)  # 归一化时间
        
        features = np.concatenate([
            start_pos / 8.0,                    # [2] 起始相对位置
            trajectory_direction,               # [2] 轨迹主方向
            [trajectory_length / 8.0],          # [1] 轨迹长度
            [curvature],                        # [1] 轨迹曲率
            [min_distance / 8.0],               # [1] 最近距离
            [closest_time],                     # [1] 最近时刻
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


    def _get_robot2_joint_indices(self):
        """根据您的XML获取robot2相关的所有关节索引"""
        
        # 🤖 根据您的XML文件定义的robot2关节名称
        robot2_joint_names = [
            # 底盘主关节（自由关节）
            "robot2:centroid",
            
            # 车轮驱动关节
            "robot2:r-l-drive-hinge",        # 后左轮驱动
            "robot2:r-r-drive-hinge",        # 后右轮驱动
            "robot2:f-l-drive-hinge-1",      # 前左轮驱动1
            "robot2:f-l-drive-hinge-2",      # 前左轮驱动2
            "robot2:f-r-drive-hinge-1",      # 前右轮驱动1
            "robot2:f-r-drive-hinge-2",      # 前右轮驱动2
            
            # 转向关节
            "robot2:ghost-steer-hinge",      # 虚拟转向
            "robot2:f-l-steer-hinge",        # 前左轮转向
            "robot2:f-r-steer-hinge",        # 前右轮转向
            
            # 机械臂关节
            "robot2:Joint1",                 # 机械臂关节1
            "robot2:Joint2",                 # 机械臂关节2
            "robot2:Joint3",                 # 机械臂关节3
            "robot2:Joint4",                 # 机械臂关节4
            "robot2:Joint5",                 # 机械臂关节5
            # 注意：Joint6在XML中被注释掉了，所以不包含
        ]
        
        joint_indices = {
            'qpos': [],
            'qvel': [],
            'ctrl': []
        }
        
        print("🔍 Identifying robot2 joint indices from XML...")
        
        for joint_name in robot2_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                
                # 获取qpos索引范围
                qpos_adr = self.model.jnt_qposadr[joint_id]
                joint_type = self.model.jnt_type[joint_id]
                
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # 自由关节：7个qpos (位置3 + 四元数4)
                    qpos_indices = list(range(qpos_adr, qpos_adr + 7))
                    qvel_indices = list(range(self.model.jnt_dofadr[joint_id], self.model.jnt_dofadr[joint_id] + 6))
                    print(f"  ✅ {joint_name} (FREE): qpos[{qpos_adr}:{qpos_adr+7}], qvel[{self.model.jnt_dofadr[joint_id]}:{self.model.jnt_dofadr[joint_id]+6}]")
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    # 铰链关节：1个qpos
                    qpos_indices = [qpos_adr]
                    qvel_indices = [self.model.jnt_dofadr[joint_id]]
                    print(f"  ✅ {joint_name} (HINGE): qpos[{qpos_adr}], qvel[{self.model.jnt_dofadr[joint_id]}]")
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    # 滑动关节：1个qpos
                    qpos_indices = [qpos_adr]
                    qvel_indices = [self.model.jnt_dofadr[joint_id]]
                    print(f"  ✅ {joint_name} (SLIDE): qpos[{qpos_adr}], qvel[{self.model.jnt_dofadr[joint_id]}]")
                else:
                    print(f"  ⚠️ {joint_name}: Unknown joint type {joint_type}")
                    continue
                
                joint_indices['qpos'].extend(qpos_indices)
                joint_indices['qvel'].extend(qvel_indices)
                
            except Exception as e:
                print(f"  ❌ {joint_name}: Not found ({e})")
                continue

        robot2_actuator_names = [
            "robot2:Joint1",              
            "robot2:Joint2",
            "robot2:Joint3", 
            "robot2:Joint4",
            "robot2:Joint5",
            "robot2:drive",               
            "robot2:ghost-steer"          
        ]
        
        print("\n🔍 Identifying robot2 actuator indices...")
        for actuator_name in robot2_actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                joint_indices['ctrl'].append(actuator_id)
                print(f"  ✅ {actuator_name}: actuator_id={actuator_id}")
            except Exception as e:
                print(f"  ❌ {actuator_name}: Not found ({e})")
                continue
        
        print(f"\n🎯 Robot2 joint mapping summary:")
        print(f"   qpos indices: {len(joint_indices['qpos'])} elements - {joint_indices['qpos']}")
        print(f"   qvel indices: {len(joint_indices['qvel'])} elements - {joint_indices['qvel']}")
        print(f"   ctrl indices: {len(joint_indices['ctrl'])} elements - {joint_indices['ctrl']}")
    
        return joint_indices

    def _store_robot2_initial_states(self):

        mujoco.mj_forward(self.model, self.data)
        
        # 存储qpos
        for idx in self.robot2_joint_indices['qpos']:
            if idx < len(self.data.qpos):
                self.robot2_initial_qpos[idx] = self.data.qpos[idx]
            else:
                print(f"  ⚠️ qpos index {idx} out of range")
        
        # 存储qvel
        for idx in self.robot2_joint_indices['qvel']:
            if idx < len(self.data.qvel):
                self.robot2_initial_qvel[idx] = self.data.qvel[idx]
            else:
                print(f"  ⚠️ qvel index {idx} out of range")
        
        # 存储ctrl
        for idx in self.robot2_joint_indices['ctrl']:
            if idx < len(self.data.ctrl):
                self.robot2_initial_ctrl[idx] = self.data.ctrl[idx]
            else:
                print(f"  ⚠️ ctrl index {idx} out of range")


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

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()