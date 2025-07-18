import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from utils.mujoco_state_loader import load_mujoco_state_from_file, restore_mujoco_state

ACTION_SPACE_REDUCTION = 10

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

        # self._check_robot2_rover_velocity()

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
            "pickingplace:table0",
            "pickingplace:table2"
        ]

        self.robot_arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.robot2_arm_bodies]
        
        self.robot2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")

        self.left_object_position = [1, -2.5, 0.29696562]
        self.right_object_position = [-1, -2.5, 0.29696562]

        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()

        self.active_joint_id = None

        if left_joint_id is not None:
            self.active_position = left_position
            self.active_joint_id = left_joint_id
            self.side = "left"
        elif right_joint_id is not None:
            self.active_position = right_position
            self.active_joint_id = right_joint_id
            self.side = "right"

        self.initial_position = None

        self.previous_center_distance = None

        self.progress_reward_scale = 1000.0  
        self.distance_threshold = 0.005 

        self.success_counter = 0
        self.success_required_steps = 5

        self.task_stage = "approach"  # approach -> alignment -> suction -> grasp -> lift
        self.stage_start_step = 0
        
        self.approach_threshold = 0.0185      # 4cm内进入对齐阶段
        self.alignment_threshold = 0.01      # 1cm内进入吸附阶段
        self.grasp_threshold = 0.01        # 1cm内认为抓取成功
        # self.suction_threshold = 0.002      # 2mm内可以激活吸附
        # self.suction_height_threshold = 0.002  # 高度2mm内可以激活吸附
        self.suction_height_threshold = 0.01  # 高度2mm内可以激活吸附
        # self.suction_radius_threshold = 0.005  # 半径5mm内可以激活吸附
        self.suction_radius_threshold = 0.015  # 半径15cm内可以激活吸附
        self.lift_height_low_bound = 0.04             # 提升4cm认为完成
        self.lift_height_high_bound = 0.08             # 提升8cm认为完成

        self.is_approaching_reward_given = False
        self.is_alignment_reward_given = False
        
        # 🎯 奖励权重
        self.reward_weights = {
            "time_penalty": 0.1,
            "distance": 1000.0,
            "alignment": 10.0,
            "suction": 200.0,
            "stability": 100.0,
            "rover_penalty": 200.0,
            "collision_penalty": 20.0,
            "moved_penalty": 300.0,
            "dropped_penalty": 20.0,
            "stage_completion": 2000.0,
            "final_completion": 2000.0
        }

        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.low_bounds = np.array([-1.0, -1.919, -0.611, -1.565, -3.142, -1.8], dtype=np.float32)
        self.high_bounds = np.array([1.0, 2.792, 1.222, 1.40, 3.142, 2.2], dtype=np.float32)

        num_actuators = self.model.nu

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )
        
        # 🎯 状态记录
        self.initial_object_height = None
        self.suction_activated = False
        self.grasp_stable_steps = 0
        self.required_grasp_steps = 10

        self.lift_stable_steps = 0
        self.required_lift_steps = 5

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        self.current_step = 0

        self.task_stage = "approach"
        self.is_approaching_reward_given = False
        self.is_alignment_reward_given = False
        self.stage_start_step = 0
        self.suction_activated = False
        self.grasp_stable_steps = 0
        self.initial_object_height = None

        self.lift_stable_steps = 0
        
        self.previous_center_distance = None
        mujoco.mj_forward(self.model, self.data)

        self._record_initial_object_height()

        return self._get_obs(), {}

    def _record_initial_object_height(self):
        """记录初始物体高度"""
        left_joint_id, left_position, right_joint_id, right_position = self._get_placed_object_info()
        
        if left_joint_id is not None:
            self.initial_object_height = left_position[2]
            self.initial_position = left_position.copy()
        elif right_joint_id is not None:
            self.initial_object_height = right_position[2]
            self.initial_position = right_position.copy()
        else:
            self.initial_object_height = None
            self.initial_position = None

    def _get_obs(self):
        robot_2_arm_positions = np.array([self.data.xpos[body_id] for body_id in self.robot_arm_ids])  # [9, 3]
        robot_2_arm_velocities = np.array([self.data.cvel[body_id] for body_id in self.robot_arm_ids])  # [9, 6]

        # get active position according to the active joint
        if self.active_joint_id is not None:
            body_id = self.model.jnt_bodyid[self.active_joint_id]
            active_position = self.data.xpos[body_id]

        object_position = active_position.copy()

        target_position = object_position.copy()
        approaching_target_position = target_position.copy()
        approaching_target_position[2] += 0.0385

        alignment_target_position = target_position.copy()
        alignment_target_position[2] += 0.0085

        # 🎯 获取当前状态信息
        approaching_sphere_info = self._get_sphere_center_to_target_info(approaching_target_position)
        approaching_current_center_distance = approaching_sphere_info['center_to_target_distance']
        approaching_center_to_target_rel = approaching_sphere_info['center_to_target_rel']

        alignment_sphere_info = self._get_sphere_center_to_target_info(alignment_target_position)
        alignment_current_center_distance = alignment_sphere_info['center_to_target_distance']
        alignment_center_to_target_rel = alignment_sphere_info['center_to_target_rel']
        
        # 🎯 计算角度信息
        approaching_center_to_target_angle_xy = np.arctan2(approaching_center_to_target_rel[1], approaching_center_to_target_rel[0])
        approaching_center_to_target_angle_z = np.arctan2(approaching_center_to_target_rel[2], np.linalg.norm(approaching_center_to_target_rel[:2]))
        
        alignment_center_to_target_angle_xy = np.arctan2(alignment_center_to_target_rel[1], alignment_center_to_target_rel[0])
        alignment_center_to_target_angle_z = np.arctan2(alignment_center_to_target_rel[2], np.linalg.norm(alignment_center_to_target_rel[:2]))
        
        # 🎯 添加圆柱体检测信息
        sphere_center = approaching_sphere_info['sphere_center']
        
        height_diff = abs(sphere_center[2] - alignment_target_position[2])
        xy_distance = np.sqrt((sphere_center[0] - alignment_target_position[0])**2 + 
                            (sphere_center[1] - alignment_target_position[1])**2)
        
        height_ok = height_diff <= self.suction_height_threshold
        radius_ok = xy_distance <= self.suction_radius_threshold
        cylinder_ready = float(height_ok and radius_ok)
        
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
        
        # 🎯 获取真空吸嘴朝向
        robot2_vacuum_quat = self.data.xquat[vacuum_sphere_body_id]

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

        adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
        adhere_control = self.data.ctrl[adhere_actuator_id]
        
        joint1_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint1")
        joint1_control_raw = self.data.ctrl[joint1_actuator_id]
        joint1_control = (joint1_control_raw - (-1.919)) / (2.792 - (-1.919)) * 2 - 1
        
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

        observation = np.concatenate([
            robot2_pos / max_position,                                # [3] - 机器人位置
            [robot2_orientation / np.pi],                             # [1] - 机器人朝向
            
            vacuum_sphere_pos / max_position,                         # [3] - vacuum sphere位置
            vacuum_sphere_vel[:3] / max_speed,                        # [3] - vacuum sphere线速度
            
            [adhere_control],                                         # [1] - 吸附控制
            [joint1_control],                                         # [1] - 关节1控制
            [joint2_control],                                         # [1] - 关节2控制
            [joint3_control],                                         # [1] - 关节3控制
            [joint4_control],                                         # [1] - 关节4控制
            [joint5_control],                                         # [1] - 关节5控制

            # 🎯 接近阶段信息
            approaching_center_to_target_rel / max_distance,          # [3] - 接近目标相对位置
            [approaching_current_center_distance / max_distance],     # [1] - 接近目标距离
            [approaching_center_to_target_angle_xy / np.pi],          # [1] - 接近水平角度
            [approaching_center_to_target_angle_z / np.pi],           # [1] - 接近垂直角度

            # 🎯 对齐阶段信息
            alignment_center_to_target_rel / max_distance,            # [3] - 对齐目标相对位置
            [alignment_current_center_distance / max_distance],       # [1] - 对齐目标距离
            [alignment_center_to_target_angle_xy / np.pi],            # [1] - 对齐水平角度
            [alignment_center_to_target_angle_z / np.pi],             # [1] - 对齐垂直角度

            # 🎯 圆柱体检测信息
            [height_diff / max_distance],                             # [1] - 高度差
            [xy_distance / max_distance],                             # [1] - XY平面距离
            [cylinder_ready],                                         # [1] - 是否在圆柱体内

            # 🎯 任务阶段信息
            [1.0 if self.task_stage == "approach" else 0.0],         # [1] - 接近阶段
            [1.0 if self.task_stage == "alignment" else 0.0],        # [1] - 对齐阶段
            [1.0 if self.task_stage == "lift" else 0.0],             # [1] - 提升阶段

            # 🎯 真空吸嘴朝向信息
            robot2_vacuum_quat,                                       # [4] - 真空吸嘴四元数
            
            vacuum_to_target_rel / max_distance,                      # [3] - vacuum body到目标相对位置
            [vacuum_to_target_distance / max_distance],               # [1] - vacuum body到目标距离
            [vacuum_to_target_angle_xy / np.pi],                      # [1] - 水平角度
            [vacuum_to_target_angle_z / np.pi],                       # [1] - 垂直角度

            np.array(arm_to_target_rels).flatten() / max_distance,    # [27] - 所有手臂到目标相对位置
            np.array(arm_to_target_distances) / max_distance,         # [9] - 所有手臂到目标距离
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

    def _get_sphere_center_to_target_info(self, target_position):
        sphere_center, sphere_positions = self._get_vacuum_sphere_center()
        
        center_to_target_rel = target_position - sphere_center
        center_to_target_distance = np.linalg.norm(center_to_target_rel)
        # print(f"Center to target distance: {center_to_target_distance:.4f}")
        # print(f"Center to target relative position: {target_position} - {sphere_center} = {center_to_target_rel}")
        
        # center_to_target_angle_xy = np.arctan2(center_to_target_rel[1], center_to_target_rel[0])
        # center_to_target_angle_z = np.arctan2(center_to_target_rel[2], np.linalg.norm(center_to_target_rel[:2]))
        
        # sphere_to_target_distances = []
        # for sphere_pos in sphere_positions:
        #     distance = np.linalg.norm(target_position - sphere_pos)
        #     sphere_to_target_distances.append(distance)
        
        # sphere_to_target_distances = np.array(sphere_to_target_distances)
        
        return {
            'sphere_center': sphere_center,
            'sphere_positions': sphere_positions,
            'center_to_target_rel': center_to_target_rel,
            'center_to_target_distance': center_to_target_distance,
            # 'center_to_target_angle_xy': center_to_target_angle_xy,
            # 'center_to_target_angle_z': center_to_target_angle_z,
            # 'sphere_to_target_distances': sphere_to_target_distances,
            # 'max_sphere_distance': np.max(sphere_to_target_distances),
            # 'min_sphere_distance': np.min(sphere_to_target_distances),
            # 'avg_sphere_distance': np.mean(sphere_to_target_distances)
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

        normalized_action = np.clip(action, -1, 1)
        real_action = self.low_bounds + (normalized_action + 1) * (self.high_bounds - self.low_bounds) / 2
        
        terminated = False
        truncated = False
        
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        reward, reached, collision_detected, dropped = self._calculate_reward()
    
        # # 🎯 每50步打印一次阶段信息
        # if self.current_step % 50 == 0:
        #     stage_info = self._get_task_stage_info()
        #     print(f"📊 步骤 {self.current_step}: 阶段={stage_info['task_stage']}, "
        #         f"持续时间={stage_info['stage_duration']}, 奖励={reward:.2f}")
        
        if reached:
            terminated = True
        
        if collision_detected:
            print("Robot2 collision with forbidden area detected, terminating episode.")
            terminated = True
        
        if dropped:
            print("Object dropped, terminating episode.")
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
        body_id = self.model.jnt_bodyid[self.active_joint_id]
        object_position = self.data.xpos[body_id]

        target_position = object_position.copy()
        approaching_target_position = target_position.copy()
        approaching_target_position[2] += 0.0485

        # 🎯 获取当前状态信息
        approaching_sphere_info = self._get_sphere_center_to_target_info(approaching_target_position)
        approaching_current_center_distance = approaching_sphere_info['center_to_target_distance']

        alignment_target_position = target_position.copy()
        alignment_target_position[2] += 0.0085
        
        # 🎯 初始化奖励
        total_reward = 0.0
        task_completed = False

        # print(f"Active position: {object_position}, Initial height: {self.initial_position}")

        dropped = False
        
        # 🎯 阶段管理
        task_stage_progress_reward = self._update_task_stage(approaching_current_center_distance, alignment_target_position)

        total_reward += task_stage_progress_reward

        time_penalty = 0.0
        
        # 🎯 阶段特定奖励
        if self.task_stage == "approach":
            if not np.allclose(self.initial_position, object_position, atol=0.1):
                print("⚠️ Object moved, applying penalty.")
                # print(f"Active position: {active_position}, Initial height: {self.initial_object_height}")
                total_reward += -self.reward_weights["moved_penalty"]
                dropped = True

            distance_reward = self._calculate_distance_reward(approaching_current_center_distance)
            self.previous_center_distance = approaching_current_center_distance
            total_reward += distance_reward

            time_penalty = -2.0 * self.reward_weights["time_penalty"]
            
        elif self.task_stage == "alignment":
            if not np.allclose(self.initial_position, object_position, atol=0.1):
                print("⚠️ Object moved, applying penalty.")
                # print(f"Active position: {active_position}, Initial height: {self.initial_object_height}")
                total_reward += -self.reward_weights["moved_penalty"]
                dropped = True

            alignment_current_center_distance = self._get_sphere_center_to_target_info(alignment_target_position)['center_to_target_distance']

            distance_reward = self._calculate_distance_reward(alignment_current_center_distance)
            self.previous_center_distance = alignment_current_center_distance
            total_reward += distance_reward                

            alignment_reward = self._calculate_alignment_reward()
            total_reward += alignment_reward

            # progressive_reward = self._new_calculate_progressive_alignment_reward()
            # total_reward += progressive_reward

            time_penalty = -1.0 * self.reward_weights["time_penalty"]
            
        # elif self.task_stage == "suction":
        #     suction_reward = self._calculate_suction_reward(current_center_distance)
        #     total_reward += suction_reward
            
        # elif self.task_stage == "grasp":
        #     grasp_reward, grasp_stable = self._calculate_grasp_reward(current_center_distance)
        #     total_reward += grasp_reward
            
        #     if grasp_stable:
        #         self.task_stage = "lift"
        #         self.stage_start_step = self.current_step
                
        elif self.task_stage == "lift":
            
            if self.initial_object_height is not None and object_position[2] < 0.29696562:
                print("⚠️ Object dropped below initial height, applying penalty.")
                # print(f"Active position: {active_position}, Initial height: {self.initial_object_height}")
                total_reward += -self.reward_weights["dropped_penalty"]
                dropped = True

            lift_reward, task_completed, suction_is_not_activated = self._calculate_lift_reward(object_position)
            dropped = dropped or suction_is_not_activated
            total_reward += lift_reward
        
        # 🎯 全局惩罚
        # rover_penalty = self._calculate_rover_penalty()
        # total_reward += rover_penalty

        collision_penalty, collision_detected = self._calculate_collision_penalty()
        total_reward += collision_penalty

        total_reward += time_penalty
        
        # 🎯 更新历史状态
        
        return total_reward, task_completed, collision_detected, dropped

    def _update_task_stage(self, approaching_current_center_distance, alignment_target_position):
        """更新任务阶段"""
        # if self.task_stage == "approach" and distance < self.approach_threshold:
        #     self.task_stage = "alignment"
        #     self.stage_start_step = self.current_step
        #     print(f"🎯 进入对齐阶段 (距离: {distance:.6f}m)")
            
        # elif self.task_stage == "alignment" and distance < self.alignment_threshold:
        #     self.task_stage = "suction"
        #     self.stage_start_step = self.current_step
        #     print(f"🎯 进入吸附阶段 (距离: {distance:.6f}m)")
            
        # elif self.task_stage == "suction" and distance < self.suction_threshold:
        #     self.task_stage = "grasp"
        #     self.stage_start_step = self.current_step
        #     self.suction_activated = True
        #     print(f"🎯 进入抓取阶段 (距离: {distance:.6f}m)")
        task_stage_progress_reward = 0.0

        if self.task_stage == "approach" and approaching_current_center_distance < self.alignment_threshold:
            self.task_stage = "alignment"
            self.stage_start_step = self.current_step
            print(f"进入对齐阶段 (距离: {approaching_current_center_distance:.6f}m)")
            self.previous_center_distance = None
            if not self.is_approaching_reward_given:
                self.is_approaching_reward_given = True
                task_stage_progress_reward = self.reward_weights["stage_completion"]
                print(f"🎯 接近阶段奖励: {task_stage_progress_reward:.2f}")
        elif self.task_stage == "alignment":
            # 🎯 获取alignment_target_position和sphere_center
            # body_id = self.model.jnt_bodyid[self.active_joint_id]
            # object_position = self.data.xpos[body_id]
            
            # alignment_target_position = object_position.copy()
            # alignment_target_position[2] += 0.0085
            
            sphere_center, _ = self._get_vacuum_sphere_center()
            
            # 🎯 检查是否在圆柱体内
            # 1. 高度检查：sphere_center的z坐标与alignment_target_position的z坐标差值
            height_diff = abs(sphere_center[2] - alignment_target_position[2])
            height_ok = height_diff <= self.suction_height_threshold
            
            # 2. 半径检查：sphere_center与alignment_target_position在xy平面的距离
            xy_distance = np.sqrt((sphere_center[0] - alignment_target_position[0])**2 + 
                                (sphere_center[1] - alignment_target_position[1])**2)
            radius_ok = xy_distance <= self.suction_radius_threshold
            
            # 🎯 如果同时满足高度和半径条件，进入下一阶段
            if height_ok and radius_ok:
                self.task_stage = "lift"
                self.stage_start_step = self.current_step
                print(f"✅ 进入提升阶段!")
                if not self.is_alignment_reward_given:
                    self.is_alignment_reward_given = True
                    task_stage_progress_reward = self.reward_weights["stage_completion"]
                    print(f"🎯 对齐阶段奖励: {task_stage_progress_reward:.2f}")
        
        return task_stage_progress_reward

    def _calculate_distance_reward(self, distance):
        """基础距离奖励 - 所有阶段都适用"""
        if self.previous_center_distance is not None:
            distance_progress = self.previous_center_distance - distance
            reward = distance_progress * self.reward_weights["distance"]
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
        if self._check_robot_forbidden_collision():
            print("Robot2 collision with forbidden area detected, applying penalty.")
            return -self.reward_weights["collision_penalty"], True
        return 0.0, False

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