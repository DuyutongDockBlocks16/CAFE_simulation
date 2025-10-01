import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from utils.mujoco_state_loader import load_mujoco_state_from_file, restore_mujoco_state

ACTION_SPACE_REDUCTION = 11

class SecondRobotPlacingMuJoCoEnv(gym.Env):
    def _set_all_robot2_gear_to_01(self):
        robot2_actuators = [
            "robot2:Joint1",
            "robot2:Joint2", 
            "robot2:Joint3",
            "robot2:Joint4",
            "robot2:Joint5",
        ]
        
        for actuator_name in robot2_actuators:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                old_gear = self.model.actuator_gear[actuator_id, 0]
                self.model.actuator_gear[actuator_id, 0] = 0.1
                print(f"✅ {actuator_name}: {old_gear:.2f} -> 0.1")
            except Exception as e:
                print(f"❌ 设置{actuator_name}失败: {e}")
    
    
    def __init__(self, xml_path, state_filepath, action_repeat=1):
        super().__init__()
        self.max_steps = 32000
        self.current_step = 0
        
        self.previous_center_distance = None

        self.action_repeat = action_repeat
        if self.action_repeat > 1:
            print(f"🔄 Action Repeat enabled: {self.action_repeat}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        state_data = load_mujoco_state_from_file(state_filepath)
        
        # self._set_all_robot2_gear_to_01()
        
        self.object_joint_name = "object1:joint"
        self.object_body_name = "object1"
        self.placing_place2_high_plane_body_name = "placingplace2_high"

        self.data.qpos[:] = state_data['qpos']
        self.data.qvel[:] = state_data['qvel'] 
        self.data.ctrl[:] = state_data['ctrl']
        
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_ctrl = self.data.ctrl.copy()

        mujoco.mj_forward(self.model, self.data)
        
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)

        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name)

        self.placing_place2_high_plane_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.placing_place2_high_plane_body_name)

        self.object_initial_position = self.data.xpos[self.object_body_id].copy()
        self.initial_object_height = self.data.xpos[self.object_body_id][2]

        self.placing_place2_high_plane_body_position = self.data.xpos[self.placing_place2_high_plane_body_id].copy()
        self.placing_place_radius = 0.15

        self.required_stable_steps = 5
        self.stable_steps = 0
        
        # print all positions
        print(f"Object initial position: {self.object_initial_position}")
        print(f"Placing place2 high plane body position: {self.placing_place2_high_plane_body_position}")
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        print(f"Robot 2 position: {self.data.xpos[rover_body_id]}")

        self.target_position_final = self.placing_place2_high_plane_body_position.copy()
        self.target_position_pre_0 = self.placing_place2_high_plane_body_position.copy()
        self.target_position_pre_0[0] -= 0.4
        self.target_position_pre_0[2] -= 0.12
        self.target_position_pre_1 = self.placing_place2_high_plane_body_position.copy()
        self.target_position_pre_1[0] -= 0.2
        self.target_position_pre_1[2] += 0.1

        self.target_position_list = [
            self.target_position_pre_0,
            self.target_position_pre_1,
            self.target_position_final
        ]

        self.current_target_position = 0 # 0: pre0, 1: pre1, 2: final
        
        self.current_phase = "HORIZONTAL_ESCAPE"  # 初始阶段
        self.phase_start_step = 0
        self.previous_height = None  # 用于计算下降速度
        self.previous_horizontal_pos = None  # 用于水平稳定性检查
        
        # 🔥 阶段切换的调试信息
        self.phase_history = []  # 记录阶段切换历史

        self.escape_completed = False
        self.lifting_completed = False
        self.approach_completed = False
        
        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.low_bounds = np.array([0, -0.611, -1.565, 0.0, 0.0], dtype=np.float32)
        # self.low_bounds = np.array([-1.0, -1.919, -0.611, -1.565, -3.142, -0.2], dtype=np.float32)
        self.high_bounds = np.array([0, 1.222, 1.40, 0.0, 0.0], dtype=np.float32)

        num_actuators = self.model.nu

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )
        
        self.robot2_arm_bodies = [
            "robot2:base",          
            "robot2:base_link",     
            "robot2:link1",         
            "robot2:link2",         
            "robot2:link3",         
            "robot2:link4",         
            "robot2:link5",         
            # "robot2:link6",         
            # "robot2:vacuum_sphere"
        ]
        
        self.robot2_body_ids = []
        for body_name in self.robot2_arm_bodies:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.robot2_body_ids.append(body_id)
            except:
                continue
            
        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "placingplace1:low_plane", "placingplace1:high_plane",
            "placingplace2:low_plane", "placingplace2:high_plane",
        ]
        
    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        self.current_step = 0
        
        self.stable_steps = 0
        
        self.previous_center_distance = None
        
        self.current_target_position = 0 # 0: pre, 1: final
        
        self.current_phase = "HORIZONTAL_ESCAPE"
        self.phase_start_step = 0
        self.previous_height = None
        self.previous_horizontal_pos = None
        self.phase_history = []
        
        self.escape_completed = False
        self.lifting_completed = False
        self.approach_completed = False
        
        # 🔥 初始化高度信息
        object_pos = self.data.xpos[self.object_body_id].copy()
        self.previous_height = object_pos[2] - self.placing_place2_high_plane_body_position[2]
        self.previous_horizontal_pos = object_pos[:2].copy()
        
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}


    def _get_obs(self):
        # 🎯 基础信息
        object_pos = self.data.xpos[self.object_body_id].copy()
        
        target_position = self.target_position_list[self.current_target_position]

        # 🎯 计算contact_site到目标的信息
        object_to_target_rel = target_position - object_pos
        object_to_target_distance = np.linalg.norm(object_to_target_rel)
        object_to_target_angle_xy = np.arctan2(object_to_target_rel[1], object_to_target_rel[0])
        object_to_target_angle_z = np.arctan2(object_to_target_rel[2], np.linalg.norm(object_to_target_rel[:2]))
        
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
        joint1_control = (joint1_control_raw - (-10)) / (10 - (-10)) * 2 - 1
        
        joint2_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint2")
        joint2_control_raw = self.data.ctrl[joint2_actuator_id]
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
        
        # 🎯 归一化参数
        max_position = 3.0
        max_distance = 1.0
        max_speed = 15.0
        
        # 🔥 边缘安全信息计算
        placing_center = self.placing_place2_high_plane_body_position[:2]
        placing_z = self.placing_place2_high_plane_body_position[2]
        object_to_center_distance = np.linalg.norm(object_pos[:2] - placing_center)
        edge_distance = object_to_center_distance - self.placing_place_radius
        
        # 🔥 高度相关信息
        height_threshold = placing_z + 0.05
        object_height = object_pos[2]
        height_above_plate = object_height - placing_z
        
        # 🔥 边缘检测激活状态
        edge_detection_active = object_height <= height_threshold
        
        # 🔥 边缘安全状态
        safety_margin = 0.02
        is_safe_distance = edge_distance > safety_margin
        is_below_plate = object_height < placing_z

        if self.is_on_plane(object_pos, self.placing_place2_high_plane_body_position[:2], self.placing_place_radius, self.placing_place2_high_plane_body_position[2]):
            on_plane_flag = 1.0
        else:
            on_plane_flag = 0.0
            
        plate_center = self.placing_place2_high_plane_body_position[:2]
        plate_z = self.placing_place2_high_plane_body_position[2]
        plate_radius = self.placing_place_radius
        
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        escape_distance = horizontal_distance - plate_radius
        height_above_plate = object_pos[2] - plate_z
        
        # 🔥 阶段编码（独热编码）
        phase_encoding = [0.0, 0.0, 0.0, 0.0]  # [ESCAPE, LIFTING, APPROACH, DESCENT]
        if self.current_phase == "HORIZONTAL_ESCAPE":
            phase_encoding[0] = 1.0
        elif self.current_phase == "VERTICAL_LIFTING":
            phase_encoding[1] = 1.0
        elif self.current_phase == "HORIZONTAL_APPROACH":
            phase_encoding[2] = 1.0
        elif self.current_phase == "PRECISION_DESCENT":
            phase_encoding[3] = 1.0
        
        # 🔥 阶段进度指示器
        if self.current_phase == "HORIZONTAL_ESCAPE":
            stage_progress = min(1.0, max(0.0, escape_distance / 0.08))
        elif self.current_phase == "VERTICAL_LIFTING":
            stage_progress = min(1.0, max(0.0, height_above_plate / 0.20))
        elif self.current_phase == "HORIZONTAL_APPROACH":
            stage_progress = min(1.0, max(0.0, (0.15 - horizontal_distance) / 0.10))
        else:  # PRECISION_DESCENT
            target_height = 0.02
            stage_progress = min(1.0, max(0.0, (0.10 - abs(height_above_plate - target_height)) / 0.08))
        
        # 🔥 运动趋势信息
        if self.previous_horizontal_pos is not None:
            horizontal_velocity = np.linalg.norm(object_pos[:2] - self.previous_horizontal_pos)
            horizontal_velocity_normalized = min(1.0, horizontal_velocity / 0.02)  # 2cm/step为最大
        else:
            horizontal_velocity_normalized = 0.0
        
        if self.previous_height is not None:
            vertical_velocity = (object_pos[2] - self.placing_place2_high_plane_body_position[2]) - self.previous_height
            vertical_velocity_normalized = np.clip(vertical_velocity / 0.01, -1.0, 1.0)  # 1cm/step为最大
        else:
            vertical_velocity_normalized = 0.0
        
        # 🎯 简化的观测空间
        observation = np.concatenate([
            object_pos / max_position,                                    # [3]
            self.placing_place2_high_plane_body_position / max_position,  # [3]
            [self.placing_place_radius / max_position],                   # [1]
            
            # 控制信号 [6]
            [adhere_control, joint1_control, joint2_control, 
            joint3_control, joint4_control, joint5_control],
            
            # 目标导航信息 [6]
            object_to_target_rel / max_distance,                         # [3]
            [object_to_target_distance / max_distance],                  # [1]
            [object_to_target_angle_xy / np.pi],                         # [1]
            [object_to_target_angle_z / np.pi],                          # [1]
            
            # vacuum sphere信息 [10]
            vacuum_sphere_pos / max_position,                            # [3]
            vacuum_sphere_vel / max_speed,                               # [6]
            vacuum_sphere_quat,                                          # [7]  
            
            # 🔥 阶段管理信息 [11维]
            phase_encoding,                                              # [4] - 当前阶段独热编码
            [stage_progress],                                            # [1] - 阶段进度
            [escape_distance / 1.0],                                     # [1] - 脱离距离
            [height_above_plate / 1.0],                                  # [1] - 相对高度
            [horizontal_distance / 1.0],                                 # [1] - 水平距离
            [horizontal_velocity_normalized],                            # [1] - 水平速度
            [vertical_velocity_normalized],                              # [1] - 垂直速度
            [float(self.current_step - self.phase_start_step) / 100.0],  # [1] - 阶段持续时间
            
            [is_safe_distance],
            [is_below_plate],
            
            # 成功指示器 [1]
            [on_plane_flag],                                            # [1]
            
        ], dtype=np.float32)
        
        return observation

    def step(self, action):
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
        # real_action = np.clip(action, self.low_bounds, self.high_bounds)
        # print(f"Real Action: {real_action}")
        # print(f"Real Action: {real_action}, Normalized Action: {normalized_action}, action: {action}")
        
        terminated = False
        truncated = False
        
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        reward, reached, collision_detected, dropped = self._calculate_reward()
    
        if reached:
            terminated = True

        if collision_detected:
            print("Robot2 collision with forbidden area detected, terminating episode.")
            terminated = True
        
        if dropped:
            print("Object dropped, terminating episode.")
            terminated = True

        if self.current_step >= self.max_steps:
            print("Maximum steps reached, terminating episode.")
            truncated = True
        
        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("⚠️ QACC error detected, terminating episode.")
            truncated = True
        
        self.current_step += 1
        
        # Track cumulative reward
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0.0
        
        self.cumulative_reward += reward
        
        # info = self._get_info()
        info = {
            'cumulative_reward': self.cumulative_reward,
            'step_reward': reward,
            'current_phase': self.current_phase
        }
        
        # print(f"Step {self.current_step} | Step Reward: {reward:.2f} | Cumulative Reward: {self.cumulative_reward:.2f}")
        
        return obs, reward, terminated, truncated, info

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

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def _calculate_reward(self):
        total_reward = 0.0
        dropped = False
        
        # 🎯 获取基础信息
        object_pos = self.data.xpos[self.object_body_id].copy()
        plate_center = self.placing_place2_high_plane_body_position[:2]
        plate_z = self.placing_place2_high_plane_body_position[2]
        plate_radius = self.placing_place_radius
        
        # 🔥 更新当前阶段
        current_phase, phase_switch = self.get_current_phase(object_pos, plate_center, plate_z, plate_radius)
        
        if phase_switch:
            total_reward += 100
        
        # 🔥 阶段特定奖励
        if current_phase == "HORIZONTAL_ESCAPE":
            phase_reward = self.horizontal_escape_phase_reward(object_pos, plate_center, plate_radius)
            status = "🔄 水平脱离"
        elif current_phase == "VERTICAL_LIFTING":
            phase_reward = self.vertical_lifting_phase_reward(object_pos, plate_center, plate_z, plate_radius)
            status = "⬆️ 垂直抬升"
        elif current_phase == "HORIZONTAL_APPROACH":
            phase_reward = self.horizontal_approach_phase_reward(object_pos, plate_center, plate_z)
            status = "➡️ 水平接近"
        else:  # PRECISION_DESCENT
            phase_reward = self.precision_descent_phase_reward(object_pos, plate_center, plate_z)
            status = "⬇️ 精确下降"
            
        phase_reward = phase_reward * 1
        
        total_reward += phase_reward
        
        # 🔥 阶段完成奖励（鼓励快速通过前期阶段）
        # stage_completion_reward = self._calculate_stage_completion_reward(current_phase, object_pos, plate_center, plate_z, plate_radius)
        # total_reward += stage_completion_reward
        
        # 🔥 最终放置检测
        if self.is_on_plane(object_pos, plate_center, plate_radius, plate_z):
            total_reward = 5.0  # 重置为稳定奖励
            self.stable_steps += 1
            if self.current_step % 40 == 0:
                print(f"🎯 稳定放置: {self.stable_steps}/{self.required_stable_steps}")
        else:
            self.stable_steps = 0
        
        # 🔥 任务完成检测
        if self.stable_steps >= self.required_stable_steps:
            total_reward += 50.0
            print("🎉 任务完成！")
            return total_reward, True, False, dropped
        
        # # 🔥 基础惩罚
        # time_penalty = -0.001  # 时间惩罚，鼓励快速完成
        # total_reward += time_penalty
        
        # 🔥 掉落检测
        if self._calculate_object_dropped():
            dropped = True
            total_reward -= 50.0
            print("💥 物体掉落!")
            
        if self._object_detached_from_end_effector():
            dropped = True
            total_reward -= 50.0
            print("💥 物体脱离末端执行器!")
        
        # 🔥 更新历史信息
        self.previous_height = object_pos[2] - plate_z
        self.previous_horizontal_pos = object_pos[:2].copy()
        
        # 🔥 调试信息
        # if self.current_step % 1 == 0:
        #     horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        #     escape_distance = horizontal_distance - plate_radius
        #     height_above_plate = object_pos[2] - plate_z
            
        #     plate_thickness = 0.01 
        #     distance_to_plate_bottom = abs(height_above_plate) - plate_thickness/2
            
        #     print(f"📊 Step {self.current_step}: {status}")
        #     print(f"   阶段奖励: {phase_reward:.2f} | 完成奖励: {stage_completion_reward:.2f}")
        #     print(f"   脱离距离: {escape_distance*100:.1f}cm | 高度: {height_above_plate*100:.1f}cm")
        #     print(f"   水平距中心: {horizontal_distance*100:.1f}cm")
        #     print(f"   距离盘底: {distance_to_plate_bottom*100:.1f}cm")
        #     print(f"   当前阶段奖励: {phase_reward:.2f}")
        #     print(f"   总奖励: {total_reward:.2f}")
        
        return total_reward, False, False, dropped
    
    def _object_detached_from_end_effector(self):
        robot2_vacuum_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")

        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 🔥 获取几何体对应的body ID
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # 🔥 检查是否是rover和vacuum sphere之间的碰撞
            if ((body1_id == robot2_vacuum_id and body2_id == object_body_id) or
                (body1_id == object_body_id and body2_id == robot2_vacuum_id)):
                
                return False
        
        return True

    def _calculate_stage_completion_reward(self, current_phase, object_pos, plate_center, plate_z, plate_radius):
        """阶段完成奖励 - 鼓励快速进入下一阶段"""
        
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        escape_distance = horizontal_distance - plate_radius
        height_above_plate = object_pos[2] - plate_z
        
        completion_reward = 0.0
        
        if current_phase == "HORIZONTAL_ESCAPE":
            # 鼓励快速脱离到安全距离
            if escape_distance > 0.08:
                completion_reward = 2.0  # 已完全脱离
            elif escape_distance > 0.05:
                completion_reward = 1.0  # 接近完成
                
        elif current_phase == "VERTICAL_LIFTING":
            # 鼓励达到安全高度
            if height_above_plate > 0.18:
                completion_reward = 2.0  # 接近完成抬升
            elif height_above_plate > 0.12:
                completion_reward = 1.0  # 抬升进行中
                
        elif current_phase == "HORIZONTAL_APPROACH":
            # 鼓励接近目标中心
            if horizontal_distance < 0.05:
                completion_reward = 3.0  # 非常接近
            elif horizontal_distance < 0.08:
                completion_reward = 1.5  # 接近目标
                
        elif current_phase == "PRECISION_DESCENT":
            # 鼓励精确下降
            if 0.01 <= height_above_plate <= 0.03:
                completion_reward = 4.0  # 理想高度
            elif height_above_plate < 0.08:
                completion_reward = 2.0  # 正在下降
        
        return completion_reward
    
    def _calculate_object_dropped(self):
        object_height = self.data.xpos[self.object_body_id][2]
        
        if self.initial_object_height is not None and object_height < 0.27:
            return True

        return False
    
    def is_on_plane(self, obj_pos, plane_pos, plane_radius, plane_z, z_tol=0.05):
        # print(f"Checking if object at {obj_pos} is on plane at {plane_pos}")
        dx = obj_pos[0] - plane_pos[0]
        dy = obj_pos[1] - plane_pos[1]
        dz = obj_pos[2] - plane_z
        return (dx**2 + dy**2) <= plane_radius**2 and 0 <= dz < z_tol

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
    
    def _calculate_distance_reward(self, distance):
        """基础距离奖励 - 所有阶段都适用"""
        if self.previous_center_distance is not None:
            distance_progress = self.previous_center_distance - distance
            reward = distance_progress * 100.0  # 放大奖励效果
            return reward
        return 0.0

    def _calculate_collision_penalty(self):
        # print("Checking for collisions...")
        if self._check_robot_forbidden_collision():
            print("Robot2 collision with forbidden area detected, applying penalty.")
            return -20, True
        
        return 0.0, False
    
    def _check_vacuum_sphere_collision_with_rover_body(self):
        """检查vacuum sphere是否与rover body碰撞"""
        
        # 🔥 获取rover body ID
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        
        # 🔥 获取vacuum sphere body ID
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 🔥 获取几何体对应的body ID
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # 🔥 检查是否是rover和vacuum sphere之间的碰撞
            if ((body1_id == rover_body_id and body2_id == vacuum_sphere_body_id) or
                (body1_id == vacuum_sphere_body_id and body2_id == rover_body_id)):
                
                print(f"🚨 检测到rover与vacuum sphere碰撞!")
                return True
        
        return False

    def _get_task_stage_info(self):
        """获取任务阶段信息（用于调试）"""
        info = {
            "task_stage": self.task_stage,
            "stage_duration": self.current_step - self.stage_start_step,
            "suction_activated": self.suction_activated,
            "grasp_stable_steps": self.grasp_stable_steps
        }
        return info
        
    def horizontal_escape_phase_reward(self, object_pos, plate_center, plate_radius):
    
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        escape_distance = horizontal_distance - plate_radius
        
        plate_z = self.placing_place2_high_plane_body_position[2]
        height_above_plate = object_pos[2] - plate_z
        
        plate_thickness = 0.01 
        distance_to_plate_bottom = abs(height_above_plate) - plate_thickness/2
        
        total_reward = 0.0
        
        if escape_distance < 0.02:
            # 🔥 危险区域额外惩罚（保留一些绝对约束）
            if distance_to_plate_bottom < 0.03:  # 极度危险
                danger_penalty = -0.05
                total_reward += danger_penalty
                # print(f"   🚨 极度危险区域惩罚: {danger_penalty:.2f}")
            
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        rover_pos = self.data.xpos[rover_body_id][:2]
        
        current_distance_to_rover = np.linalg.norm(object_pos[:2] - rover_pos)
        
        if hasattr(self, 'previous_distance_to_rover') and self.previous_distance_to_rover is not None:
            rover_approach_progress = self.previous_distance_to_rover - current_distance_to_rover
            if rover_approach_progress > 0: 
                rover_direction_reward = rover_approach_progress * 80.0
            else:
                rover_direction_reward = rover_approach_progress * 80.0
            
            total_reward += rover_direction_reward
        
        # 更新小车距离历史
        self.previous_distance_to_rover = current_distance_to_rover
        self.previous_escape_distance = escape_distance
        self.previous_plate_bottom_distance = distance_to_plate_bottom  # 这里确保变量被设置
        
        return total_reward

    def vertical_lifting_phase_reward(self, object_pos, plate_center, plate_z, plate_radius):
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        escape_distance = horizontal_distance - plate_radius
        height_above_plate = object_pos[2] - plate_z
        
        total_reward = 0.0
        
        if escape_distance < 0.01:
            return -0.05

        target_safe_height = 0.06

        distance_to_target_height = abs(height_above_plate - target_safe_height)
    
        if hasattr(self, 'previous_distance_to_target_height') and self.previous_distance_to_target_height is not None:
            target_approach_progress = self.previous_distance_to_target_height - distance_to_target_height
            
            if target_approach_progress > 0:  # 接近目标高度
                target_approach_reward = target_approach_progress * 80.0
                total_reward += target_approach_reward
            else:  # 远离目标高度
                target_approach_reward = target_approach_progress * 80.0
                total_reward += target_approach_reward
            
        self.previous_distance_to_target_height = distance_to_target_height
        
        return total_reward
        
    def horizontal_approach_phase_reward(self, object_pos, plate_center, plate_z):
        height_above_plate = object_pos[2] - plate_z
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        
        min_safe_height = 0.03 
        if height_above_plate < min_safe_height:
            return -0.05 
        
        total_reward = 0.0
    
        # 🔥 水平接近进步奖励
        if hasattr(self, 'previous_horizontal_approach_distance') and self.previous_horizontal_approach_distance is not None:
            approach_progress = self.previous_horizontal_approach_distance - horizontal_distance
            
            if approach_progress > 0:  # 接近目标中心
                approach_progress_reward = approach_progress * 80.0  # 奖励接近
                total_reward += approach_progress_reward
            else:  # 远离目标中心
                approach_progress_reward = approach_progress * 80.0  # 惩罚远离
                total_reward += approach_progress_reward
        
        self.previous_horizontal_approach_distance = horizontal_distance
        return total_reward
        
    def precision_descent_phase_reward(self, object_pos, plate_center, plate_z):
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        height_above_plate = object_pos[2] - plate_z
        
        # 基础安全检查
        if horizontal_distance > 0.08:
            return -0.05 
        
        total_reward = 0.0
        target_height = 0.02  # 目标高度2cm
        
        # 🔥 高度进步奖励 - 简单版本
        if hasattr(self, 'previous_descent_height') and self.previous_descent_height is not None:
            height_progress = self.previous_descent_height - height_above_plate
            
            # 下降奖励，上升惩罚
            height_progress_reward = height_progress * 80.0
            total_reward += height_progress_reward
        
        # 🔥 更新历史数据
        self.previous_descent_height = height_above_plate
        
        return total_reward

    def get_current_phase(self, object_pos, plate_center, plate_z, plate_radius):
    
        horizontal_distance = np.linalg.norm(object_pos[:2] - plate_center)
        escape_distance = horizontal_distance - plate_radius
        height_above_plate = object_pos[2] - plate_z
        
        # 🔥 检查各阶段完成条件
        if not self.escape_completed and escape_distance > 0.05:
            self.escape_completed = True
            print("✅ 水平脱离阶段完成！")
            
        if not self.lifting_completed and height_above_plate > 0.06 and self.escape_completed:
            self.lifting_completed = True
            print("✅ 垂直抬升阶段完成！")
            
        if not self.approach_completed and horizontal_distance < 0.06 and self.lifting_completed:
            self.approach_completed = True
            print("✅ 水平接近阶段完成！")
        
        # 🔥 基于完成标志的不可逆阶段判断
        if not self.escape_completed:
            new_phase = "HORIZONTAL_ESCAPE"
            condition = f"水平脱离中，脱离距离{escape_distance*100:.1f}cm (需要>5cm)"
            
        elif not self.lifting_completed:
            new_phase = "VERTICAL_LIFTING"
            condition = f"垂直抬升中，高度{height_above_plate*100:.1f}cm (需要>12cm)"
            
        elif not self.approach_completed:
            new_phase = "HORIZONTAL_APPROACH"
            condition = f"水平接近中，距离{horizontal_distance*100:.1f}cm (需要<6cm)"
            
        else:
            new_phase = "PRECISION_DESCENT"
            condition = f"精确下降中，高度{height_above_plate*100:.1f}cm"
        
        phase_switch = False
        # 🔥 阶段切换检测和记录
        if new_phase != self.current_phase:
            phase_switch = True
            duration = self.current_step - self.phase_start_step
            print(f"🎯 阶段切换: {self.current_phase} -> {new_phase}")
            print(f"   切换原因: {condition}")
            print(f"   持续步数: {duration}")
            
            # 记录阶段历史
            self.phase_history.append({
                'from_phase': self.current_phase,
                'to_phase': new_phase,
                'step': self.current_step,
                'duration': duration,
                'condition': condition
            })
            
            self.current_phase = new_phase
            self.phase_start_step = self.current_step
        
        return new_phase, phase_switch