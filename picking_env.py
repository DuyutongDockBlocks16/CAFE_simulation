import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from utils.mujoco_state_loader import load_mujoco_state_from_file, restore_mujoco_state

ACTION_SPACE_REDUCTION = 11

class SecondRobotPickingMuJoCoEnv(gym.Env):
    def __init__(self, xml_path, state_filepath, action_repeat=4):
        super().__init__()

        self.action_repeat = action_repeat
        if self.action_repeat > 1:
            print(f"🔄 Action Repeat enabled: {self.action_repeat}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        state_data = load_mujoco_state_from_file(state_filepath)

        self.data.qpos[:] = state_data['qpos']
        self.data.qvel[:] = state_data['qvel'] 
        self.data.ctrl[:] = state_data['ctrl']
        robot1_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:drive")
        self.model.actuator_gear[robot1_motor_id] = 0.0
        # robot2_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:drive")
        # self.model.actuator_gear[robot2_motor_id] = 0.0
        # reset all data.ctrl to 0
        self.data.qfrc_applied[:] = state_data['qfrc_applied']
        self.data.time = state_data['time']

        mujoco.mj_forward(self.model, self.data)

        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        self.data.cvel[rover_body_id] = 0.0

        self._check_robot2_rover_velocity()

        # rover_joints = ["robot2:ghost-steer-hinge", "robot2:drive"]
        # for joint_name in rover_joints:
        #     joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        #     self.model.dof_damping[joint_id] = 1000.0

        time_step = 0.005
        self.model.opt.timestep = time_step

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

        self.max_steps = 5000
        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)

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
            "pickingplace:table0", "pickingplace:table1", 
            "pickingplace:table2", "pickingplace:table3",
            "object0_geom" , "object1_geom", "object2_geom", "object3_geom", "object4_geom", 
            "object5_geom", "object6_geom", "object7_geom", "object8_geom", "object9_geom"
        ]

        self.robot_arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.robot2_arm_bodies]
        
        self.robot2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")

        self.left_object_position = [1, -2.5, 0.29198]
        self.right_object_position = [-1, -2.5, 0.29198]

        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.low_bounds = np.array([-1.919, -0.611, -1.565, -3.142, -1.8], dtype=np.float32)
        self.high_bounds = np.array([2.792, 1.222, 1.40, 3.142, 2.2], dtype=np.float32)

        num_actuators = self.model.nu

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )

        self.previous_center_distance = None
        self.previous_max_deviation = None
        self.previous_avg_deviation = None

        self.progress_reward_scale = 1000.0  
        self.distance_threshold = 0.005 

        self.success_counter = 0
        self.success_required_steps = 5

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        self.current_step = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):
        robot_2_arm_positions = np.array([self.data.xpos[body_id] for body_id in self.robot_arm_ids])  # [9, 3]
        robot_2_arm_velocities = np.array([self.data.cvel[body_id] for body_id in self.robot_arm_ids])  # [9, 6]

        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        if left_joint_id is not None:
            active_position = left_position
            active_joint_id = left_joint_id
            side = "left"
        elif right_joint_id is not None:
            active_position = right_position
            active_joint_id = right_joint_id
            side = "right"

        target_position = active_position.copy()
        target_position[2] += 0.0085

        sphere_info = self._get_sphere_center_to_target_info(target_position)
        
        sphere_center = sphere_info['sphere_center']
        center_to_target_rel = sphere_info['center_to_target_rel']
        center_to_target_distance = sphere_info['center_to_target_distance']
        center_to_target_angle_xy = sphere_info['center_to_target_angle_xy']
        center_to_target_angle_z = sphere_info['center_to_target_angle_z']
        
        robot_2_arm_positions = np.array([self.data.xpos[body_id] for body_id in self.robot_arm_ids])
        robot_2_arm_velocities = np.array([self.data.cvel[body_id] for body_id in self.robot_arm_ids])
        
        arm_to_target_rels = []
        arm_to_target_distances = []
        
        for body_id in self.robot_arm_ids:
            arm_body_pos = self.data.xpos[body_id]
            arm_target_rel = target_position - arm_body_pos
            arm_to_target_rels.append(arm_target_rel)
            arm_to_target_distances.append(np.linalg.norm(arm_target_rel))
        
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_sphere_pos = self.data.xpos[vacuum_sphere_body_id]
        vacuum_sphere_vel = self.data.cvel[vacuum_sphere_body_id]

        linear_velocity = vacuum_sphere_vel[:3]
        current_max_velocity = np.linalg.norm(linear_velocity)
        
        vacuum_to_target_rel = target_position - vacuum_sphere_pos
        vacuum_to_target_distance = np.linalg.norm(vacuum_to_target_rel)
        vacuum_to_target_angle_xy = np.arctan2(vacuum_to_target_rel[1], vacuum_to_target_rel[0])
        vacuum_to_target_angle_z = np.arctan2(vacuum_to_target_rel[2], np.linalg.norm(vacuum_to_target_rel[:2]))
        
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        robot2_quat = self.data.xquat[self.robot2_rover_id]
        robot2_orientation = self._quaternion_to_yaw(robot2_quat)
        
        max_position = 3.0
        max_distance = 3.0
        max_speed = 15.0
        
        observation = np.concatenate([
            robot2_pos / max_position,                    # [3] - 机器人位置
            [robot2_orientation / np.pi],                 # [1] - 机器人朝向
            
            vacuum_sphere_pos / max_position,             # [3] - vacuum sphere位置
            vacuum_sphere_vel[:3] / max_speed,            # [3] - vacuum sphere线速度
            
            center_to_target_rel / max_distance,          # [3] - 圆心到目标相对位置
            [center_to_target_distance / max_distance],   # [1] - 圆心到目标距离
            [center_to_target_angle_xy / np.pi],          # [1] - 水平角度
            [center_to_target_angle_z / np.pi],           # [1] - 垂直角度
            
            [sphere_info['max_sphere_distance'] / max_distance],  # [1] - 最大偏差
            [sphere_info['avg_sphere_distance'] / max_distance],  # [1] - 平均偏差
            [sphere_info['min_sphere_distance'] / max_distance],  # [1] - 最小偏差
            
            vacuum_to_target_rel / max_distance,          # [3] - vacuum body到目标相对位置
            [vacuum_to_target_distance / max_distance],   # [1] - vacuum body到目标距离
            [vacuum_to_target_angle_xy / np.pi],          # [1] - 水平角度
            [vacuum_to_target_angle_z / np.pi],           # [1] - 垂直角度
            
            np.array(arm_to_target_rels).flatten() / max_distance,  # [27] - 所有手臂到目标相对位置
            np.array(arm_to_target_distances) / max_distance,       # [9] - 所有手臂到目标距离
        
        ], dtype=np.float32)
        
        return observation

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
                # print(f"✅ 左侧放置对象: object{object_id}, joint_id={joint_id}")

            elif np.allclose(position, self.right_object_position, atol=0.1):
                right_joint_id = joint_id
                right_position = position.copy()
                # print(f"✅ 右侧放置对象: object{object_id}, joint_id={joint_id}")
        
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

    def _get_sphere_center_to_target_info(self, target_position):
        sphere_center, sphere_positions = self._get_vacuum_sphere_center()
        
        center_to_target_rel = target_position - sphere_center
        center_to_target_distance = np.linalg.norm(center_to_target_rel)
        
        center_to_target_angle_xy = np.arctan2(center_to_target_rel[1], center_to_target_rel[0])
        center_to_target_angle_z = np.arctan2(center_to_target_rel[2], np.linalg.norm(center_to_target_rel[:2]))
        
        sphere_to_target_distances = []
        for sphere_pos in sphere_positions:
            distance = np.linalg.norm(target_position - sphere_pos)
            sphere_to_target_distances.append(distance)
        
        sphere_to_target_distances = np.array(sphere_to_target_distances)
        
        return {
            'sphere_center': sphere_center,
            'sphere_positions': sphere_positions,
            'center_to_target_rel': center_to_target_rel,
            'center_to_target_distance': center_to_target_distance,
            'center_to_target_angle_xy': center_to_target_angle_xy,
            'center_to_target_angle_z': center_to_target_angle_z,
            'sphere_to_target_distances': sphere_to_target_distances,
            'max_sphere_distance': np.max(sphere_to_target_distances),
            'min_sphere_distance': np.min(sphere_to_target_distances),
            'avg_sphere_distance': np.mean(sphere_to_target_distances)
        }

    def step(self, action):
        # self._check_robot2_rover_velocity()
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
        # if self.current_step % 50 == 0:  # 每50步监控一次
        #     self._monitor_arm_car_interaction()

        normalized_action = np.clip(action, -1, 1)
        real_action = self.low_bounds + (normalized_action + 1) * (self.high_bounds - self.low_bounds) / 2
        
        terminated = False
        truncated = False
        
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        reward, reached = self._calculate_reward()
        
        if reached:
            terminated = True
        
        if self._check_robot_forbidden_collision():
            print("Robot2 collision with forbidden area detected, terminating episode.")
            reward -= 20
            terminated = True
        
        if self.current_step >= self.max_steps:
            truncated = True
        
        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("⚠️ QACC error detected, terminating episode.")
            truncated = True
        
        self.current_step += 1
        
        # info = self._get_info()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()
        
        if left_joint_id is not None:
            active_position = left_position
        elif right_joint_id is not None:
            active_position = right_position
        else:
            return -1.0, False
        
        target_position = active_position.copy()
        target_position[1] += 0.0085
        
        # 🎯 获取当前状态信息
        sphere_info = self._get_sphere_center_to_target_info(target_position)
        current_center_distance = sphere_info['center_to_target_distance']
        current_max_deviation = sphere_info['max_sphere_distance']
        current_avg_deviation = sphere_info['avg_sphere_distance']
        
        # 🎯 初始化奖励
        total_reward = 0.0
        is_in_target = current_center_distance < self.distance_threshold
        
        if is_in_target:
            self.success_counter += 1
            print(f"🎯 在目标范围内! 计数器: {self.success_counter}/{self.success_required_steps}, 距离: {current_center_distance:.6f}m")
        else:
            if self.success_counter > 0:
                print(f"离开目标范围! 计数器重置, 距离: {current_center_distance:.6f}m")
            self.success_counter = 0
        
        reached = self.success_counter >= self.success_required_steps
        
        if self.previous_center_distance is not None:
            distance_progress = self.previous_center_distance - current_center_distance
            distance_reward = distance_progress * self.progress_reward_scale
            
            max_deviation_progress = self.previous_max_deviation - current_max_deviation
            avg_deviation_progress = self.previous_avg_deviation - current_avg_deviation
            precision_reward = (max_deviation_progress + avg_deviation_progress) * self.progress_reward_scale * 0.5
            
            total_reward = distance_reward + precision_reward
        
        rover_velocity_penalty = self._calculate_rover_velocity_penalty()
        total_reward += rover_velocity_penalty
        
        # 🎯 额外奖励
        if is_in_target:
            # 在目标范围内给予额外奖励
            stay_reward = 40  # 保持越久奖励越多
            total_reward += stay_reward
        
        if reached:
            print(f"🎉 任务成功完成! 最终距离: {current_center_distance:.6f}m")
        
        # 🎯 更新上一帧的状态
        self.previous_center_distance = current_center_distance
        self.previous_max_deviation = current_max_deviation
        self.previous_avg_deviation = current_avg_deviation
        
        return total_reward, reached

    def _check_task_completion(self):
        try:
            # 🎯 获取目标位置
            left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()
            
            if left_joint_id is not None:
                active_position = left_position
            elif right_joint_id is not None:
                active_position = right_position
            else:
                return False
            
            target_position = active_position.copy()
            target_position[1] += 0.0085
            
            # 🎯 检查是否达到目标
            sphere_info = self._get_sphere_center_to_target_info(target_position)
            center_distance = sphere_info['center_to_target_distance']
            
            # 🎯 任务完成条件
            completion_threshold = 0.005  # 5mm内认为完成
            return center_distance < completion_threshold
            
        except Exception as e:
            print(f"❌ 检查任务完成时发生错误: {e}")
            return False

    def _check_collision(self):
        """检查是否有碰撞"""
        # 🎯 这里可以实现具体的碰撞检测逻辑
        # 例如检查与禁止区域的碰撞
        
        # 检查所有接触点
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 获取几何体对应的body
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # 获取几何体名称
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            # 检查robot2是否与禁止区域碰撞
            if hasattr(self, 'robot2_body_ids') and hasattr(self, 'forbidden_geoms'):
                if ((body1_id in self.robot2_body_ids and geom2_name in self.forbidden_geoms) or
                    (body2_id in self.robot2_body_ids and geom1_name in self.forbidden_geoms)):
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

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def _check_robot2_rover_velocity(self):
        """检查robot2:rover的速度信息"""
        try:
            # 获取robot2:rover的body ID
            rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
            
            # 获取速度信息
            rover_vel = self.data.cvel[rover_body_id]  # 6D速度 [vx, vy, vz, wx, wy, wz]
            linear_vel = rover_vel[:3]  # 线速度
            angular_vel = rover_vel[3:]  # 角速度
            
            # 获取位置信息
            rover_pos = self.data.xpos[rover_body_id]
            rover_quat = self.data.xquat[rover_body_id]
            
            print("\n🚗 Robot2:rover 初始状态检查:")
            print("=" * 50)
            print(f"位置: [{rover_pos[0]:8.5f}, {rover_pos[1]:8.5f}, {rover_pos[2]:8.5f}]")
            print(f"四元数: [{rover_quat[0]:6.3f}, {rover_quat[1]:6.3f}, {rover_quat[2]:6.3f}, {rover_quat[3]:6.3f}]")
            print(f"线速度: [{linear_vel[0]:8.5f}, {linear_vel[1]:8.5f}, {linear_vel[2]:8.5f}]")
            print(f"角速度: [{angular_vel[0]:8.5f}, {angular_vel[1]:8.5f}, {angular_vel[2]:8.5f}]")
            
            # 计算速度大小
            linear_speed = np.linalg.norm(linear_vel)
            angular_speed = np.linalg.norm(angular_vel)
            
            print(f"线速度大小: {linear_speed:.6f} m/s")
            print(f"角速度大小: {angular_speed:.6f} rad/s")
            
            # 判断是否静止
            if linear_speed < 1e-6 and angular_speed < 1e-6:
                print("✅ Robot2:rover 处于静止状态")
            else:
                print("⚠️ Robot2:rover 存在初始速度!")
                if linear_speed > 1e-6:
                    print(f"   线速度不为零: {linear_speed:.6f} m/s")
                if angular_speed > 1e-6:
                    print(f"   角速度不为零: {angular_speed:.6f} rad/s")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ 检查robot2:rover速度时出错: {e}")


    def _monitor_arm_car_interaction(self):
        """监控机械手臂和小车的相互作用"""
        # 获取小车速度
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        rover_vel = self.data.cvel[rover_body_id]
        rover_linear_vel = rover_vel[:3]
        rover_angular_vel = rover_vel[3:]
        
        # 获取机械手臂末端速度
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        arm_vel = self.data.cvel[vacuum_sphere_body_id]
        arm_linear_vel = arm_vel[:3]
        
        # 计算速度大小
        rover_speed = np.linalg.norm(rover_linear_vel)
        arm_speed = np.linalg.norm(arm_linear_vel)
        
        # 如果机械手臂在运动，检查小车是否也在运动
        if arm_speed > 1e-4:  # 机械手臂在运动
            if rover_speed > 1e-6:  # 小车也在运动
                print(f"🔗 手臂-小车相互作用:")
                print(f"   手臂速度: {arm_speed:.6f} m/s")
                print(f"   小车速度: {rover_speed:.6f} m/s")
                print(f"   速度比: {rover_speed/arm_speed:.4f}")
                
                # 检查动量方向是否相反
                arm_momentum = arm_linear_vel
                rover_momentum = rover_linear_vel
                
                # 计算动量的点积（负值表示方向相反）
                momentum_dot = np.dot(arm_momentum, rover_momentum)
                if momentum_dot < 0:
                    print("   ✅ 动量方向相反（符合物理定律）")
                else:
                    print("   ⚠️ 动量方向相同（可能有其他力作用）")

    def _calculate_rover_velocity_penalty(self):
        """计算小车速度惩罚"""
        try:
            rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
            rover_vel = self.data.cvel[rover_body_id]
            
            linear_vel = rover_vel[:3]
            angular_vel = rover_vel[3:]
            
            linear_speed = np.linalg.norm(linear_vel)
            angular_speed = np.linalg.norm(angular_vel)
            
            # 🎯 设置惩罚系数
            linear_penalty_scale = -0.5  # 线速度惩罚系数
            angular_penalty_scale = -0.1  # 角速度惩罚系数
            
            # 🎯 计算惩罚
            linear_penalty = linear_speed * linear_penalty_scale
            angular_penalty = angular_speed * angular_penalty_scale
            
            total_penalty = linear_penalty + angular_penalty
            
            # 🎯 只在有显著速度时打印
            # if linear_speed > 1e-4 or angular_speed > 1e-4:
            #     print(f"🚗 小车速度惩罚: 线速度={linear_speed:.6f}m/s, 角速度={angular_speed:.6f}rad/s, 惩罚={total_penalty:.3f}")
            
            return total_penalty
            
        except Exception as e:
            print(f"❌ 计算小车速度惩罚时出错: {e}")
            return 0.0