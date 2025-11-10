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


        # robot1_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:drive")
        # self.model.actuator_gear[robot1_motor_id] = 0.0
        robot2_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:drive")
        self.model.actuator_gear[robot2_motor_id] = 0.0
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

        self.max_steps = 2000
        # self.max_steps = 5000
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
        
        self.object_geoms = [
            "object0_geom", "object1_geom", "object2_geom", "object3_geom",
            "object4_geom", "object5_geom", "object6_geom", "object7_geom",
            "object8_geom", "object9_geom"
        ]
        
        self.vacuum_sphere_body = ["robot2:vacuum_sphere"]

        self.vacuum_sphere_body_ids = []
        for body_name in self.vacuum_sphere_body:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.vacuum_sphere_body_ids.append(body_id)
            except:
                continue

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

        self.progress_reward_scale = 2000.0  
        self.distance_threshold = 0.005 

        self.success_counter = 0
        self.success_required_steps = 5

        self.task_stage = "approach"  # approach -> alignment -> suction -> grasp -> lift
        self.stage_start_step = 0
        
        self.approach_threshold = 0.0185      # Enter alignment stage within 4cm
        self.alignment_threshold = 0.01      # Enter suction stage within 1cm
        self.grasp_threshold = 0.01        # Consider grasp successful within 1cm
        # self.suction_threshold = 0.002      # Can activate suction within 2mm
        # self.suction_height_threshold = 0.002  # Can activate suction within 2mm height
        self.suction_height_threshold = 0.01  # Can activate suction within 2mm height
        # self.suction_radius_threshold = 0.005  # Can activate suction within 5mm radius
        self.suction_radius_threshold = 0.015  # Can activate suction within 15cm radius
        self.lift_height_low_bound = 0.04             # Consider completion when lifted 4cm
        self.lift_height_high_bound = 0.08             # Consider completion when lifted 8cm

        self.is_approaching_reward_given = False
        self.is_alignment_reward_given = False
        
        # 🎯 Reward weights
        self.reward_weights = {
            "time_penalty": 0.1,
            "distance": 1000.0,
            "alignment": 10.0,
            "suction": 200.0,
            "speed_penalty": 1.0,
            "stability": 100.0,
            "rover_penalty": 200.0,
            "collision_penalty": 20.0,
            "moved_penalty": 20.0,
            "dropped_penalty": 200.0,
            "stage_completion": 2000.0,
            "final_completion": 2000.0
        }

        obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.low_bounds = np.array([-1.0, -10, 0, -1.565, -3.142, -0.2], dtype=np.float32)
        self.high_bounds = np.array([1.0, 10, 1.222, 1.40, 3.142, 0.2], dtype=np.float32)
        
        
        # self.low_bounds = np.array([-1.0, -1.919, -0.611, -1.565, -3.142, -0.2], dtype=np.float32)
        # self.high_bounds = np.array([1.0, 2.792, 1.222, 1.40, 3.142, 0.2], dtype=np.float32)

        num_actuators = self.model.nu

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )
        
        # 🎯 State tracking
        self.initial_object_height = None
        self.suction_activated = False
        self.grasp_stable_steps = 0
        self.required_grasp_steps = 10

        self.lift_stable_steps = 0
        self.required_lift_steps = 5

        self.picking_stable_steps = 0

    def reset(self, seed=None, options=None):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        self.current_step = 0
        
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        print(f"in Rover position: {self.data.xpos[rover_body_id]}")

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
        self.picking_stable_steps = 0

        self._record_initial_object_height()

        return self._get_obs(), {}

    def _record_initial_object_height(self):
        """Record initial object height"""
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
        
        # 🎯 Basic information
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        robot2_quat = self.data.xquat[self.robot2_rover_id]
        robot2_orientation = self._quaternion_to_yaw(robot2_quat)
        
        # 🎯 Get target position
        if self.active_joint_id is not None:
            body_id = self.model.jnt_bodyid[self.active_joint_id]
            target_position = self.data.xpos[body_id]
        
        vacuum_contact_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot2:vacuum_contact_site")
        vacuum_contact_site_pos = self.data.site_xpos[vacuum_contact_site_id]
        # print(f"✅ Using vacuum_contact_site position: {vacuum_contact_site_pos}")

        # 🎯 Calculate contact_site to target information
        contact_to_target_rel = target_position - vacuum_contact_site_pos
        contact_to_target_distance = np.linalg.norm(contact_to_target_rel)
        contact_to_target_angle_xy = np.arctan2(contact_to_target_rel[1], contact_to_target_rel[0])
        contact_to_target_angle_z = np.arctan2(contact_to_target_rel[2], np.linalg.norm(contact_to_target_rel[:2]))
        
        # 🎯 vacuum_sphere velocity and orientation information
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_sphere_pos = self.data.xpos[vacuum_sphere_body_id]
        vacuum_sphere_vel = self.data.cvel[vacuum_sphere_body_id]
        vacuum_sphere_quat = self.data.xquat[vacuum_sphere_body_id]
        
        # 🎯 Control signals
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
        
        # 🎯 Sensor data
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "robot2:vacuum_touch")
        sensor_data = self.data.sensordata[sensor_id]
        if sensor_data > 0 and self._check_robot_object_collision():
            sensor_data = 1.0
        
        # 🎯 Normalization parameters
        max_position = 3.0
        max_distance = 1.0
        max_speed = 15.0
        
        # 🎯 Simplified observation space
        observation = np.concatenate([
            # Basic position and orientation information
            robot2_pos / max_position,                                    # [3] - robot position
            [robot2_orientation / np.pi],                                 # [1] - robot orientation
            
            # vacuum contact point position and velocity
            vacuum_contact_site_pos / max_position,                       # [3] - contact site position
            vacuum_sphere_vel[:3] / max_speed,                           # [3] - vacuum sphere linear velocity
            
            # Control signals
            [adhere_control],                                            # [1] - suction control
            [joint1_control],                                            # [1] - joint1 control
            [joint2_control],                                            # [1] - joint2 control
            [joint3_control],                                            # [1] - joint3 control
            [joint4_control],                                            # [1] - joint4 control
            [joint5_control],                                            # [1] - joint5 control
            [sensor_data],                                               # [1] - contact sensor
            
            # 🎯 Core: contact_site to target information
            contact_to_target_rel / max_distance,                        # [3] - relative position
            [contact_to_target_distance / max_distance],                 # [1] - distance
            [contact_to_target_angle_xy / np.pi],                        # [1] - horizontal angle
            [contact_to_target_angle_z / np.pi],                         # [1] - vertical angle
            
            # vacuum sphere orientation (for alignment task)
            vacuum_sphere_quat,                                          # [4] - orientation quaternion
            
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
            [0.0, 0.0, 0.0],        # center sphere
            [0.0027, 0.0027, 0.0],  # top-right
            [0.0027, -0.0027, 0.0], # bottom-right
            [-0.0027, 0.0027, 0.0], # top-left
            [-0.0027, -0.0027, 0.0], # bottom-left
            [-0.003, 0.0, 0.0],     # left
            [0.003, 0.0, 0.0],      # right
            [0.0, -0.003, 0.0],     # bottom
            [0.0, 0.003, 0.0]       # top
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
        # print the action
        # print(f"Action: {action}")

        if self.action_repeat == 1:
            # If action_repeat=1, use original step logic
            return self._original_step(action)
        else:
            # If action_repeat>1, repeat action execution
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
    
        # # 🎯 Print stage information every 50 steps
        # if self.current_step % 50 == 0:
        #     stage_info = self._get_task_stage_info()
        #     print(f"📊 Step {self.current_step}: Stage={stage_info['task_stage']}, "
        #         f"Duration={stage_info['stage_duration']}, Reward={reward:.2f}")
        
        if reached:
            terminated = True
        
        # if is_speeding:
        #     print("Robot2 is speeding, applying speed penalty.")
        #     terminated = True

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
        """Simplified reward function - using vacuum_contact_site"""
        
        # 🎯 Get target position
        body_id = self.model.jnt_bodyid[self.active_joint_id]
        target_position = self.data.xpos[body_id]
        

        vacuum_contact_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot2:vacuum_contact_site")
        vacuum_contact_site_pos = self.data.site_xpos[vacuum_contact_site_id]

        
        # 🎯 Calculate contact_site to target distance
        contact_to_target_distance = np.linalg.norm(target_position - vacuum_contact_site_pos)
        
        # 🎯 Initialize reward
        total_reward = 0.0
        dropped = False
        
        # 🎯 Distance reward - core driving force
        distance_reward = max(self._calculate_distance_reward(contact_to_target_distance), -100)
        total_reward += distance_reward
        self.previous_center_distance = contact_to_target_distance
        
        # 🎯 Contact reward
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "robot2:vacuum_touch")
        sensor_data = self.data.sensordata[sensor_id]
        touched = False
        
        if sensor_data > 0 and self._check_robot_object_collision():
            print("✅ Vacuum touch sensor activated, object touched.")
            touched = True
            total_reward += 200.0
        else:
            # 🎯 Only check if object moved when not in contact
            if not np.allclose(self.initial_position, target_position, atol=0.1):
                print("⚠️ Object moved, applying penalty.")
                total_reward += -self.reward_weights["moved_penalty"]
                dropped = True
        
        # 🎯 Suction control detection
        adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
        adhere_control = self.data.ctrl[adhere_actuator_id]
        suction_activated = (adhere_control == 1.0)
        
        # 🎯 Grasp success detection
        picked = touched and suction_activated
        if picked:
            self.picking_stable_steps += 1
            picking_reward = self.reward_weights["suction"]
            total_reward += picking_reward
            print("✅ Suction successful, object attached.")
        else:
            if self.picking_stable_steps > 0:
                print("❌ Suction failed, object not attached.")
                dropped = True
            # self.picking_stable_steps = 0
            
        
        # 🎯 Task completion detection
        task_completed = False
        if self.picking_stable_steps >= 10:
            task_completed = True
            total_reward += self.reward_weights["final_completion"]
            print("✅ Task completed! Object successfully attached.")
        
        # 🎯 Speed penalty - prevent overly aggressive actions
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_vel = self.data.cvel[vacuum_sphere_body_id][:3]
        speed = np.linalg.norm(vacuum_vel)
        
        # is_speeding = speed > 0.5
        # if is_speeding:
        #     total_reward += -self.reward_weights["speed_penalty"] * speed
        if speed > 0.5:  # If speed exceeds 0.5m/s, apply penalty
            speed_penalty = -1.0 * self.reward_weights["speed_penalty"] * speed
            total_reward += speed_penalty
        
        
        # 🎯 Orientation reward - only important when distance is close enough
        if contact_to_target_distance < 0.05:  # Consider orientation only within 5cm
            alignment_reward = self._calculate_alignment_reward()
            total_reward += alignment_reward
        
        # 🎯 Time penalty
        time_penalty = -1.0 * self.reward_weights["time_penalty"]
        total_reward += time_penalty
        
        # 🎯 Collision detection
        collision_penalty, collision_detected = self._calculate_collision_penalty()
        # print(f"Collision detected: {collision_detected}, applying penalty: {collision_penalty:.2f}")
        total_reward += collision_penalty
        
        # 🎯 Debug information
        # if self.current_step % 100 == 0:
        #     print(f"🎯 Reward information (Step {self.current_step}):")
        #     print(f"   Contact distance: {contact_to_target_distance:.6f}m")
        #     print(f"   Distance reward: {distance_reward:.2f}")
        #     print(f"   Total reward: {total_reward:.2f}")
        #     print(f"   Contact status: {'✅' if touched else '❌'}")
        #     print(f"   Suction status: {'✅' if suction_activated else '❌'}")
        
        return total_reward, task_completed, collision_detected, dropped
    
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

    def _update_task_stage(self, approaching_current_center_distance, alignment_target_position):
        """Update task stage"""
        # if self.task_stage == "approach" and distance < self.approach_threshold:
        #     self.task_stage = "alignment"
        #     self.stage_start_step = self.current_step
        #     print(f"🎯 Enter alignment stage (distance: {distance:.6f}m)")
            
        # elif self.task_stage == "alignment" and distance < self.alignment_threshold:
        #     self.task_stage = "suction"
        #     self.stage_start_step = self.current_step
        #     print(f"🎯 Enter suction stage (distance: {distance:.6f}m)")
            
        # elif self.task_stage == "suction" and distance < self.suction_threshold:
        #     self.task_stage = "grasp"
        #     self.stage_start_step = self.current_step
        #     self.suction_activated = True
        #     print(f"🎯 Enter grasp stage (distance: {distance:.6f}m)")
        task_stage_progress_reward = 0.0

        if self.task_stage == "approach" and approaching_current_center_distance < self.alignment_threshold:
            self.task_stage = "alignment"
            self.stage_start_step = self.current_step
            print(f"Enter alignment stage (distance: {approaching_current_center_distance:.6f}m)")
            self.previous_center_distance = None
            if not self.is_approaching_reward_given:
                self.is_approaching_reward_given = True
                task_stage_progress_reward = self.reward_weights["stage_completion"]
                print(f"🎯 Approaching stage reward: {task_stage_progress_reward:.2f}")
        elif self.task_stage == "alignment":
            # 🎯 Get alignment_target_position and sphere_center
            # body_id = self.model.jnt_bodyid[self.active_joint_id]
            # object_position = self.data.xpos[body_id]
            
            # alignment_target_position = object_position.copy()
            # alignment_target_position[2] += 0.0085
            
            sphere_center, _ = self._get_vacuum_sphere_center()
            
            # 🎯 Check if within cylinder
            # 1. Height check: z-coordinate difference between sphere_center and alignment_target_position
            height_diff = abs(sphere_center[2] - alignment_target_position[2])
            height_ok = height_diff <= self.suction_height_threshold
            
            # 2. Radius check: distance between sphere_center and alignment_target_position in xy plane
            xy_distance = np.sqrt((sphere_center[0] - alignment_target_position[0])**2 + 
                                (sphere_center[1] - alignment_target_position[1])**2)
            radius_ok = xy_distance <= self.suction_radius_threshold
            
            # 🎯 If both height and radius conditions are met, enter next stage
            if height_ok and radius_ok:
                self.task_stage = "lift"
                self.stage_start_step = self.current_step
                print(f"✅ Enter lift stage!")
                if not self.is_alignment_reward_given:
                    self.is_alignment_reward_given = True
                    task_stage_progress_reward = self.reward_weights["stage_completion"]
                    print(f"🎯 Alignment stage reward: {task_stage_progress_reward:.2f}")
        
        return task_stage_progress_reward

    def _calculate_distance_reward(self, distance):
        """Basic distance reward - applicable to all stages"""
        if self.previous_center_distance is not None:
            distance_progress = self.previous_center_distance - distance
            reward = distance_progress * self.reward_weights["distance"]
            return reward
        return 0.0

    def _calculate_approach_reward(self, distance):
        # """Approach stage reward"""
        # # 🎯 Higher reward for closer distance
        # approach_reward = -distance * 5
        
        # # 🎯 Moderate speed reward
        # vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        # vacuum_vel = self.data.cvel[vacuum_sphere_body_id][:3]
        # speed = np.linalg.norm(vacuum_vel)
        
        # if distance > 0.05: 
        #     optimal_speed = 0.1  
        #     speed_reward = -abs(speed - optimal_speed) * 1
        # else:  # Need to slow down when approaching
        #     speed_reward = -speed * 5
        
        # return approach_reward + speed_reward
        return 0.0

    def _calculate_alignment_reward(self):
        """Alignment stage reward"""
        standard_quaternions = np.array([-0.707, 0.0, 0.707, 0.0])

        robot2_vacuum_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        robot2_vacuum_quat = self.data.xquat[robot2_vacuum_body_id]

        orientation_similarity = np.abs(np.dot(robot2_vacuum_quat, standard_quaternions))
        
        
        # 🎯 Extended precision reward tiers
        if orientation_similarity > 0.995:    # Within ~5 degrees - perfect alignment
            precision_reward = 1.0 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.99:   # Within ~8 degrees - excellent alignment
            precision_reward = 0.5 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.98:   # Within ~11 degrees - good alignment
            precision_reward = 0.2 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.95:   # Within ~18 degrees - acceptable alignment
            precision_reward = 0.1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.9:    # Within ~26 degrees - fair alignment
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.8:    # Within ~37 degrees - poor alignment
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.7:    # Within ~46 degrees - bad alignment
            precision_reward = -1 * self.reward_weights["alignment"]
        elif orientation_similarity > 0.5:    # Within ~60 degrees - very bad alignment
            precision_reward = -1 * self.reward_weights["alignment"]
        else:                                  # Greater than 60 degrees - worst alignment
            precision_reward = -1.0 * self.reward_weights["alignment"]

        # print(f"Alignment precision reward: {precision_reward:.3f}, robot2_vacuum_quat: {robot2_vacuum_quat}")

        return precision_reward 

    def _calculate_suction_reward(self, distance):
        """Suction stage reward"""
        # 🎯 Activate suction when distance is close enough
        if distance < self.suction_threshold:
            suction_reward = 100.0
            
            try:
                adhere_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
                adhere_control = self.data.ctrl[adhere_actuator_id]
                
                if adhere_control > 0.6:  # Suction activated
                    suction_reward += 50.0
                    print(f"🔗 Vacuum suction activated: {adhere_control:.3f}")
                else:
                    print(f"⚠️ Vacuum suction not activated: {adhere_control:.3f}")
                    
            except Exception as e:
                print(f"❌ Error checking vacuum suction control signal: {e}")
                
        else:
            suction_reward = -distance * 0.2  # Penalty for being too far
        
        # 🎯 Stability reward
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        vacuum_vel = self.data.cvel[vacuum_sphere_body_id][:3]
        speed = np.linalg.norm(vacuum_vel)
        
        stability_reward = -speed * 0.1  # Need stability during suction
        
        return suction_reward + stability_reward

    def _calculate_grasp_reward(self, distance):
        """Grasp stage reward"""
        grasp_reward = 0.0
        
        # 🎯 Maintain within grasp range
        if distance < self.grasp_threshold:
            self.grasp_stable_steps += 1
            grasp_reward += 20.0  # Reward for maintaining each step

            if adhere_control > 0.6:  # Suction activated
                suction_reward += 50.0
            
            # 🎯 Continuous stability reward
            if self.grasp_stable_steps >= self.required_grasp_steps:
                grasp_reward += 200.0
                grasp_stable = True
            else:
                grasp_stable = False
                
            print(f"🤏 Grasp stability: {self.grasp_stable_steps}/{self.required_grasp_steps}")
        else:
            # 🎯 Reset when leaving grasp range
            if self.grasp_stable_steps > 0:
                print(f"❌ Grasp failed, resetting counter")
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

        if adhere_control == 1.0:  # Suction activated
            suction_reward += 1.0
            print(f"🔗 Vacuum suction activated: {adhere_control:.3f}")

            current_height = object_position[2]
            lift_height = current_height - 0.29696562
        
            lift_reward = min(lift_height * 100.0, 8.0)
            print(f"Lift height: {lift_height:.6f}m, reward: {lift_reward:.2f}")
            if lift_reward > 0.0:
                suction_reward += 1.0
                print(f"🔗 Vacuum suction activated and object lifted: {adhere_control:.3f}")

            # Check if maintaining lift height
            if self.lift_height_low_bound < lift_height and lift_height < self.lift_height_high_bound:
                self.lift_stable_steps += 1
                # lift_reward += 2.0  # Reward for maintaining lift each step
                lift_reward += 10.0  # Reward for maintaining lift each step
                
                # Continuous stable lift reward
                if self.lift_stable_steps >= self.required_lift_steps:
                    completion_reward = self.reward_weights["final_completion"]
                    task_completed = True
                    print(f"Task completed! Lift height: {lift_height:.6f}m, stable steps: {self.lift_stable_steps}")
                else:
                    completion_reward = 0.0
                    task_completed = False
                    print(f"Lift stability: {self.lift_stable_steps}/{self.required_lift_steps} (height: {lift_height:.6f}m)")
            elif lift_height > self.lift_height_high_bound:
                print(f"Lift height exceeds range (height: {lift_height:.6f}m)")
                suction_reward = 0.0
                lift_reward = 0.0
                self.lift_stable_steps = 0
                completion_reward = 0.0
                task_completed = False
            else:
                # Insufficient height, reset counter
                if self.lift_stable_steps > 0:
                    print(f"Insufficient lift height or exceeds range, resetting counter (height: {lift_height:.6f}m)")
                self.lift_stable_steps = 0
                completion_reward = 0.0
                task_completed = False

        else:
            suction_is_not_activated = True
            print("❌ Vacuum suction not activated, cannot lift object")
            suction_reward -= 1.0
            print(f"⚠️ Vacuum suction not activated: {adhere_control:.3f}")
        
        return lift_reward + completion_reward + suction_reward, task_completed, suction_is_not_activated

    def _calculate_collision_penalty(self):
        # print("Checking for collisions...")
        if self._check_robot_forbidden_collision() or self._check_vacuum_sphere_collision_with_rover_body():
            print("Robot2 collision with forbidden area detected, applying penalty.")
            return -self.reward_weights["collision_penalty"], True
        
        return 0.0, False
    
    def _check_vacuum_sphere_collision_with_rover_body(self):
        """Check if vacuum sphere collides with rover body"""
        
        # 🔥 Get rover body ID
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        
        # 🔥 Get vacuum sphere body ID
        vacuum_sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 🔥 Get body IDs corresponding to geometries
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # 🔥 Check if collision is between rover and vacuum sphere
            if ((body1_id == rover_body_id and body2_id == vacuum_sphere_body_id) or
                (body1_id == vacuum_sphere_body_id and body2_id == rover_body_id)):
                
                print(f"🚨 Collision detected between rover and vacuum sphere!")
                return True
        
        return False

    def _get_suction_direction(self, quat):
        rotation_matrix = self._quaternion_to_rotation_matrix(quat)
        local_direction = np.array([0, 0, -1])
        world_direction = rotation_matrix @ local_direction
        return world_direction

    def _get_task_stage_info(self):
        """Get task stage information (for debugging)"""
        info = {
            "task_stage": self.task_stage,
            "stage_duration": self.current_step - self.stage_start_step,
            "suction_activated": self.suction_activated,
            "grasp_stable_steps": self.grasp_stable_steps
        }
        return info

    def _new_calculate_progressive_alignment_reward(self):
        """Alignment stage reward - encourage both approaching and aligning"""
        
        # 🎯 Orientation reward
        standard_quaternions = np.array([-0.707, 0.0, 0.707, 0.0])
        robot2_vacuum_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:vacuum_sphere")
        robot2_vacuum_quat = self.data.xquat[robot2_vacuum_body_id]
        orientation_similarity = np.abs(np.dot(robot2_vacuum_quat, standard_quaternions))
        
        # 🎯 Position information
        body_id = self.model.jnt_bodyid[self.active_joint_id]
        object_position = self.data.xpos[body_id]
        alignment_target_position = object_position.copy()
        alignment_target_position[2] += 0.0085
        
        sphere_center, _ = self._get_vacuum_sphere_center()
        
        # 🎯 Key improvement: calculate height and XY distance separately
        height_diff = abs(sphere_center[2] - alignment_target_position[2])
        xy_distance = np.sqrt((sphere_center[0] - alignment_target_position[0])**2 + 
                            (sphere_center[1] - alignment_target_position[1])**2)
        
        # 🎯 Height reward (vertical alignment)
        height_reward = max(0, (self.suction_height_threshold * 3 - height_diff) / (self.suction_height_threshold * 3)) * 20
        
        # 🎯 XY position reward (horizontal alignment)
        xy_reward = max(0, (self.suction_radius_threshold * 3 - xy_distance) / (self.suction_radius_threshold * 3)) * 20
        
        # 🎯 Orientation reward (but only give high reward when distance is close enough)
        distance_factor = max(0, 1.0 - (height_diff + xy_distance) / 0.02)  # Factor increases as distance decreases
        
        if orientation_similarity > 0.995:
            orientation_reward = 30.0 * (1.0 + distance_factor)  # Double orientation reward at close distance
        elif orientation_similarity > 0.99:
            orientation_reward = 15.0 * (1.0 + distance_factor * 0.5)
        elif orientation_similarity > 0.98:
            orientation_reward = 8.0
        elif orientation_similarity > 0.95:
            orientation_reward = 3.0
        else:
            orientation_reward = -5.0  # Penalty for poor orientation
        
        # 🎯 Combined reward: all three dimensions are important
        total_alignment_reward = (height_reward + xy_reward + orientation_reward) * self.reward_weights["alignment"] / 100.0
        
        # 🎯 Debug information
        if self.current_step % 100 == 0:
            print(f"🎯 Alignment reward breakdown:")
            print(f"   Orientation similarity: {orientation_similarity:.4f} -> reward: {orientation_reward:.2f}")
            print(f"   Height diff: {height_diff*1000:.1f}mm -> reward: {height_reward:.2f}")
            print(f"   XY distance: {xy_distance*1000:.1f}mm -> reward: {xy_reward:.2f}")
            print(f"   Distance factor: {distance_factor:.3f}")
            print(f"   Total alignment reward: {total_alignment_reward:.2f}")
        
        return total_alignment_reward