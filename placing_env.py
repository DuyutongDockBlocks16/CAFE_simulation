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
    
    
    def __init__(self, xml_path, state_filepath, action_repeat=4):
        super().__init__()
        self.max_steps = 8000
        
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

        self.required_stable_steps = 20
        self.stable_steps = 0
        
        # print all positions
        print(f"Object initial position: {self.object_initial_position}")
        print(f"Placing place2 high plane body position: {self.placing_place2_high_plane_body_position}")

        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.low_bounds = np.array([0, 0, -1.565, 0.0, 0.0], dtype=np.float32)
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
            
        self.forbidden_geoms = [
            "wall_front", "wall_back", "wall_left", "wall_right",
            "pickingplace:table0",
            "pickingplace:table2"
        ]

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        self.current_step = 0
        
        self.stable_steps = 0
        
        self.previous_center_distance = None
        
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}


    def _get_obs(self):
        # 🎯 基础信息
        object_pos = self.data.xpos[self.object_body_id].copy()
        
        target_position = self.placing_place2_high_plane_body_position.copy()
        target_position[2] += 0.05
        target_position[0] -= 0.05

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
        joint2_control = (joint2_control_raw - (0)) / (1.222 - (0)) * 2 - 1
        # joint2_control = (joint2_control_raw - (-0.611)) / (1.222 - (-0.611)) * 2 - 1
        
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
        
        # 🎯 简化的观测空间
        observation = np.concatenate([
            # 基础位置和朝向信息
            object_pos / max_position,                                    # [3] - 机器人位置
            
            # 控制信号
            [adhere_control],                                            # [1] - 吸附控制
            [joint1_control],                                            # [1] - 关节1控制
            [joint2_control],                                            # [1] - 关节2控制
            [joint3_control],                                            # [1] - 关节3控制
            [joint4_control],                                            # [1] - 关节4控制
            [joint5_control],                                            # [1] - 关节5控制

            # 🎯 核心：object到目标的信息
            object_to_target_rel / max_distance,                        # [3] - 相对位置
            [object_to_target_distance / max_distance],                 # [1] - 距离
            [object_to_target_angle_xy / np.pi],                        # [1] - 水平角度
            [object_to_target_angle_z / np.pi],                         # [1] - 垂直角度

            # vacuum sphere朝向（用于对齐任务）
            vacuum_sphere_quat,                                          # [4] - 朝向四元数
            
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
        
        # info = self._get_info()
        info = {}
        
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
        """简化的奖励函数 - 使用vacuum_contact_site"""
        
        # 🎯 获取目标位置
        object_pos = self.data.xpos[self.object_body_id].copy()
        target_position = self.placing_place2_high_plane_body_position.copy()
        target_position[2] += 0.05
        
        # 🎯 计算contact_site到目标的距离
        object_to_target_distance = np.linalg.norm(target_position - object_pos)
        
        # 🎯 初始化奖励
        total_reward = 0.0
        dropped = False
        
        # 🎯 距离奖励 - 核心驱动力
        distance_reward = max(self._calculate_distance_reward(object_to_target_distance), -100)
        total_reward += distance_reward
        self.previous_center_distance = object_to_target_distance
        
        # if object is on plane
        if self.is_on_plane(object_pos, self.placing_place2_high_plane_body_position[:2], self.placing_place_radius, self.placing_place2_high_plane_body_position[2]):
            total_reward = 0.0
            self.stable_steps += 1
            total_reward += 10.0
        else:
            self.stable_steps = 0
        
        # 🎯 任务完成检测
        task_completed = False
        if self.stable_steps >= self.required_stable_steps:
            task_completed = True
            total_reward += 50
            print("✅ 任务完成！物体已成功吸附。")
        
        # 🎯 时间惩罚
        time_penalty = -0.005
        total_reward += time_penalty
        
        # 🎯 碰撞检测
        collision_penalty, collision_detected = self._calculate_collision_penalty()
        # print(f"Collision detected: {collision_detected}, applying penalty: {collision_penalty:.2f}")
        total_reward += collision_penalty
        
        object_dropped = self._calculate_object_dropped()
        if object_dropped:
            dropped = True
            total_reward -= 20
            
        return total_reward, task_completed, collision_detected, dropped
    
    def _calculate_object_dropped(self):
        object_height = self.data.xpos[self.object_body_id][2]
        
        if self.initial_object_height is not None and object_height < self.initial_object_height - 0.05:
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

    def _calculate_approach_reward(self, distance):
        # """接近阶段奖励"""
        # # 🎯 距离越近奖励越高
        # approach_reward = -distance * 5
        
        # # 🎯 适中速度奖励
        # vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        # vacuum_vel = self.data.cvel[vacuum_sphere_body_id][:3]
        # speed = np.linalg.norm(vacuum_vel)
        
        # if distance > 0.05: 
        #     optimal_speed = 0.1  
        #     speed_reward = -abs(speed - optimal_speed) * 1
        # else:  # 接近时需要减速
        #     speed_reward = -speed * 5
        
        # return approach_reward + speed_reward
        return 0.0

    def _calculate_alignment_reward(self):
        """对齐阶段奖励"""
        standard_quaternions = np.array([-0.707, 0.0, 0.707, 0.0])

        robot2_vacuum_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        robot2_vacuum_quat = self.data.xquat[robot2_vacuum_body_id]

        orientation_similarity = np.abs(np.dot(robot2_vacuum_quat, standard_quaternions))
        
        
        # 🎯 扩展的精度奖励阶梯
        if orientation_similarity > 0.995:    # 约5度内 - 完美对齐
            precision_reward = 1.0 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.99:   # 约8度内 - 优秀对齐
            precision_reward = 0.5 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.98:   # 约11度内 - 良好对齐
            precision_reward = 0.2 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.95:   # 约18度内 - 可接受对齐
            precision_reward = 0.1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.9:    # 约26度内 - 一般对齐
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.8:    # 约37度内 - 较差对齐
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.7:    # 约46度内 - 差对齐
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.5:    # 约60度内 - 很差对齐
            precision_reward = -1 * self.reward_weights["alignment"]
        else:                                  # 大于60度 - 最差对齐
            precision_reward = -1.0 * self.reward_weights["alignment"]

        # print(f"对齐精度奖励: {precision_reward:.3f}, robot2_vacuum_quat: {robot2_vacuum_quat}")

        return precision_reward 

    def _calculate_suction_reward(self, distance):
        """吸附阶段奖励"""
        # 🎯 距离足够近时激活吸附
        if distance < self.suction_threshold:
            suction_reward = 100.0
            
            try:
                adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
                adhere_control = self.data.ctrl[adhere_actuator_id]
                
                if adhere_control > 0.6:  # 吸附激活
                    suction_reward += 50.0
                    print(f"🔗 真空吸附激活: {adhere_control:.3f}")
                else:
                    print(f"⚠️ 真空吸附未激活: {adhere_control:.3f}")
                    
            except Exception as e:
                print(f"❌ 检查真空吸附控制信号时出错: {e}")
                
        else:
            suction_reward = -distance * 0.2  # 距离太远的惩罚
        
        # 🎯 稳定性奖励
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_vel = self.data.cvel[vacuum_sphere_body_id][:3]
        speed = np.linalg.norm(vacuum_vel)
        
        stability_reward = -speed * 0.1  # 吸附时需要稳定
        
        return suction_reward + stability_reward

    def _calculate_grasp_reward(self, distance):
        """抓取阶段奖励"""
        grasp_reward = 0.0
        
        # 🎯 保持在抓取范围内
        if distance < self.grasp_threshold:
            self.grasp_stable_steps += 1
            grasp_reward += 20.0  # 每步保持奖励

            if adhere_control > 0.6:  # 吸附激活
                suction_reward += 50.0
            
            # 🎯 连续稳定奖励
            if self.grasp_stable_steps >= self.required_grasp_steps:
                grasp_reward += 200.0
                grasp_stable = True
            else:
                grasp_stable = False
                
            print(f"🤏 抓取稳定: {self.grasp_stable_steps}/{self.required_grasp_steps}")
        else:
            # 🎯 离开抓取范围重置
            if self.grasp_stable_steps > 0:
                print(f"❌ 抓取失败，重置计数器")
            self.grasp_stable_steps = 0
            grasp_reward -= 50.0
            grasp_stable = False
        
        return grasp_reward, grasp_stable

    def _calculate_lift_reward(self, object_position):

        task_completed = False
        completion_reward = 0.0
        lift_reward = 0.0
        suction_reward = 0.0

        adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
        adhere_control = self.data.ctrl[adhere_actuator_id]

        suction_is_not_activated = False

        if adhere_control == 1.0:  # 吸附激活
            suction_reward += 1.0
            print(f"🔗 真空吸附激活: {adhere_control:.3f}")

            current_height = object_position[2]
            lift_height = current_height - 0.29696562
        
            lift_reward = min(lift_height * 100.0, 8.0)
            print(f"提升高度: {lift_height:.6f}m, 奖励: {lift_reward:.2f}")
            if lift_reward > 0.0:
                suction_reward += 1.0
                print(f"🔗 真空吸附激活,且物体提升: {adhere_control:.3f}")

            # 检查是否保持在提升高度
            if self.lift_height_low_bound < lift_height and lift_height < self.lift_height_high_bound:
                self.lift_stable_steps += 1
                # lift_reward += 2.0  # 每步保持提升奖励
                lift_reward += 10.0  # 每步保持提升奖励
                
                # 连续稳定提升奖励
                if self.lift_stable_steps >= self.required_lift_steps:
                    completion_reward = self.reward_weights["final_completion"]
                    task_completed = True
                    print(f"任务完成! 提升高度: {lift_height:.6f}m, 稳定步数: {self.lift_stable_steps}")
                else:
                    completion_reward = 0.0
                    task_completed = False
                    print(f"提升稳定: {self.lift_stable_steps}/{self.required_lift_steps} (高度: {lift_height:.6f}m)")
            elif lift_height > self.lift_height_high_bound:
                print(f"提升高度超出范围 (高度: {lift_height:.6f}m)")
                suction_reward = 0.0
                lift_reward = 0.0
                self.lift_stable_steps = 0
                completion_reward = 0.0
                task_completed = False
            else:
                # 高度不足，重置计数器
                if self.lift_stable_steps > 0:
                    print(f"提升高度不足或者超出，重置计数器 (高度: {lift_height:.6f}m)")
                self.lift_stable_steps = 0
                completion_reward = 0.0
                task_completed = False

        else:
            suction_is_not_activated = True
            print("❌ 真空吸附未激活，无法提升物体")
            suction_reward -= 1.0
            print(f"⚠️ 真空吸附未激活: {adhere_control:.3f}")
        
        return lift_reward + completion_reward + suction_reward, task_completed, suction_is_not_activated

    def _calculate_collision_penalty(self):
        # print("Checking for collisions...")
        if self._check_robot_forbidden_collision() or self._check_vacuum_sphere_collision_with_rover_body():
            print("Robot2 collision with forbidden area detected, applying penalty.")
            return -self.reward_weights["collision_penalty"], True
        
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

    def _get_suction_direction(self, quat):
        rotation_matrix = self._quaternion_to_rotation_matrix(quat)
        local_direction = np.array([0, 0, -1])
        world_direction = rotation_matrix @ local_direction
        return world_direction

    def _get_task_stage_info(self):
        """获取任务阶段信息（用于调试）"""
        info = {
            "task_stage": self.task_stage,
            "stage_duration": self.current_step - self.stage_start_step,
            "suction_activated": self.suction_activated,
            "grasp_stable_steps": self.grasp_stable_steps
        }
        return info

    def _new_calculate_progressive_alignment_reward(self):
        """对齐阶段奖励 - 同时鼓励接近和对正"""
        
        # 🎯 朝向奖励
        standard_quaternions = np.array([-0.707, 0.0, 0.707, 0.0])
        robot2_vacuum_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        robot2_vacuum_quat = self.data.xquat[robot2_vacuum_body_id]
        orientation_similarity = np.abs(np.dot(robot2_vacuum_quat, standard_quaternions))
        
        # 🎯 位置信息
        body_id = self.model.jnt_bodyid[self.active_joint_id]
        object_position = self.data.xpos[body_id]
        alignment_target_position = object_position.copy()
        alignment_target_position[2] += 0.0085
        
        sphere_center, _ = self._get_vacuum_sphere_center()
        
        # 🎯 关键改进：分别计算高度和XY距离
        height_diff = abs(sphere_center[2] - alignment_target_position[2])
        xy_distance = np.sqrt((sphere_center[0] - alignment_target_position[0])**2 + 
                            (sphere_center[1] - alignment_target_position[1])**2)
        
        # 🎯 高度奖励（垂直对齐）
        height_reward = max(0, (self.suction_height_threshold * 3 - height_diff) / (self.suction_height_threshold * 3)) * 20
        
        # 🎯 XY位置奖励（水平对齐）
        xy_reward = max(0, (self.suction_radius_threshold * 3 - xy_distance) / (self.suction_radius_threshold * 3)) * 20
        
        # 🎯 朝向奖励（但要求距离足够近才给高奖励）
        distance_factor = max(0, 1.0 - (height_diff + xy_distance) / 0.02)  # 距离越近因子越大
        
        if orientation_similarity > 0.995:
            orientation_reward = 30.0 * (1.0 + distance_factor)  # 近距离时朝向奖励翻倍
        elif orientation_similarity > 0.99:
            orientation_reward = 15.0 * (1.0 + distance_factor * 0.5)
        elif orientation_similarity > 0.98:
            orientation_reward = 8.0
        elif orientation_similarity > 0.95:
            orientation_reward = 3.0
        else:
            orientation_reward = -5.0  # 朝向太差给惩罚
        
        # 🎯 组合奖励：三个维度都重要
        total_alignment_reward = (height_reward + xy_reward + orientation_reward) * self.reward_weights["alignment"] / 100.0
        
        # 🎯 调试信息
        if self.current_step % 100 == 0:
            print(f"🎯 对齐奖励分解:")
            print(f"   朝向相似度: {orientation_similarity:.4f} -> 奖励: {orientation_reward:.2f}")
            print(f"   高度差: {height_diff*1000:.1f}mm -> 奖励: {height_reward:.2f}")
            print(f"   XY距离: {xy_distance*1000:.1f}mm -> 奖励: {xy_reward:.2f}")
            print(f"   距离因子: {distance_factor:.3f}")
            print(f"   总对齐奖励: {total_alignment_reward:.2f}")
        
        return total_alignment_reward