import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController 
import threading
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
from config.env_config import FiniteState
from enum import IntEnum

class Robot2Action(IntEnum):
    MOVE_TO_TARGET = 0      # 移动到目标位置
    BRAKE = 1              # 制动
    ABORT_MOVEMENT = 2     # 舍弃当前移动
    GRASP = 3              # 抓取
    PLACE = 4              # 放置

class Robot2State(IntEnum):
    IDLE = 0               # 空闲
    MOVING = 1             # 移动中
    BRAKING = 2            # 制动中
    GRASPING = 3           # 抓取中
    PLACING = 4            # 放置中

class SecondRobotMuJoCoEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        time_step = 0.005
        self.model.opt.timestep = time_step

        # ...existing object setup code...
        object_ids = self.get_object_ids(self.model)
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

        # Start the asynchronous thread
        self.start_object_remover_threads(self.model, self.data, self.object_joint_ids)

        self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")
        
        self.robot2_joint1_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint1")
        self.robot2_joint2_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint2")
        self.robot2_joint3_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint3")
        self.robot2_joint4_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint4")
        self.robot2_joint5_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:Joint5")
        self.robot2_adhere_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:adhere_winch")
        self.robot2_ghost_steer_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:ghost-steer")
        self.robot2_drive_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:drive")
        self.robot2_steer_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot2:steer")
        
        self.target_position_x_y = [-1, -1.7] 
        
        # 🎯 Robot2状态管理
        self.robot2_state = Robot2State.IDLE
        self.robot2_target_pos = None
        self.robot2_prev_pos_error = np.zeros(2)
        self.robot2_prev_yaw_error = 0.0
        self.robot2_movement_start_time = 0
        self.robot2_max_movement_time = 200

        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(len(Robot2Action))

        self.max_steps = 8000
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
            "robot1:rover", "robot1:r-l-wheel", "robot1:r-r-wheel", 
            "robot1:f-l-wheel", "robot1:f-l-wheel-1", "robot1:f-l-wheel-2",
            "robot1:f-r-wheel", "robot1:f-r-wheel-1", "robot1:f-r-wheel-2",
            "robot1:base", "robot1:base_link", "robot1:link1", "robot1:link2",
            "robot1:link3", "robot1:link4", "robot1:link5", "robot1:link6",
            "vacuum_sphere"
        ]

        self.robot2_bodies = [
            "robot2:rover", "robot2:r-l-wheel", "robot2:r-r-wheel",
            "robot2:f-l-wheel", "robot2:f-l-wheel-hub", "robot2:f-l-wheel-1", "robot2:f-l-wheel-2",
            "robot2:f-r-wheel-hub", "robot2:f-r-wheel-1", "robot2:f-r-wheel-2", "robot2:f-r-wheel",
            "robot2:base", "robot2:base_link", "robot2:link1", "robot2:link2",
            "robot2:link3", "robot2:link4", "robot2:link5", "robot2:link6"
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

    def reset(self, seed=None, options=None):
        self.target_position_x_y = random.choice(self.target_positions)

        if self.finished:       
            if self.shared_state["stop"] is False:
                self.shared_state["stop"] = True
                # while self.shared_state["stopped"] is False:
                #     pass
                self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": False}
 
            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = self.initial_qvel
            self.data.ctrl[:] = self.initial_ctrl

            self.start_object_placer_thread(self.model, self.data, self.object_joint_ids, self.left_object_position, self.right_object_position, self.shared_state)
            self.start_object_remover_threads(self.model, self.data, self.object_joint_ids)
            self.first_robot_controller.set_state(FiniteState.IDLE)
            self.first_robot_controller.reset_all_joints()

            self.current_world_step = 0

            mujoco.mj_forward(self.model, self.data)
            self.current_world_step = 0
            self.finished = False

        if not self.finished:
            self.reset_robot2_only()

        self.current_step = 0

        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id]
        self.prev_dist = np.linalg.norm(robot_2_rover_pos[0:2] - self.target_position_x_y)

        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        status = self.first_robot_controller.get_status()
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            terminated = True

        self._execute_robot2_action(action)
        
        self._update_robot2_pid_control()

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        robot_2_rover_pos = self.data.xpos[self.robot_2_rover_id][:2]
        reward, reached = self.reward_function(robot_2_rover_pos)

        truncated = False
        if self.check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 20000
            truncated = True

        if self.check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 20000
            truncated = True

        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("QACC error detected! Simulation unstable, exiting loop.")
            truncated = True

        if reached:
            print("Robot2 has reached the target area! Terminating episode.")
            terminated = True

        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 10000
        
        info = {
            'robot2_state': self.robot2_state.name,
            'robot2_target_pos': self.robot2_target_pos,
            'action_executed': Robot2Action(action).name
        }
        return obs, reward, terminated, truncated, info

    def _execute_robot2_action(self, action):
        action_type = Robot2Action(action)
        
        if action_type == Robot2Action.MOVE_TO_TARGET:
            self._start_movement_to_target()
            
        elif action_type == Robot2Action.BRAKE:
            self._brake_robot2()
            
        elif action_type == Robot2Action.ABORT_MOVEMENT:
            self._abort_movement()
            
        elif action_type == Robot2Action.GRASP:
            self._grasp_object()
            
        elif action_type == Robot2Action.PLACE:
            self._place_object()

    def _start_movement_to_target(self):
        if self.robot2_state in [Robot2State.IDLE, Robot2State.BRAKING]:
            self.robot2_state = Robot2State.MOVING
            self.robot2_target_pos = np.array(self.target_position_x_y)
            self.robot2_target_yaw = self.target_orientation
            self.robot2_movement_start_time = self.current_step
            print(f"🎯 Robot2 starting movement to {self.robot2_target_pos}")

    def _brake_robot2(self):
        self.robot2_state = Robot2State.BRAKING
        self.data.ctrl[self.robot2_drive_index] = 0
        self.data.ctrl[self.robot2_steer_index] = 0

    def _abort_movement(self):
        if self.robot2_state == Robot2State.MOVING:
            self.robot2_state = Robot2State.IDLE
            self.robot2_target_pos = None
            self.data.ctrl[self.robot2_drive_index] = 0
            self.data.ctrl[self.robot2_steer_index] = 0

    def _grasp_object(self):
        if self.robot2_state == Robot2State.IDLE:
            self.robot2_state = Robot2State.GRASPING
            print("🤏 Robot2 grasping")
            # TODO: 实现实际的抓取逻辑

    def _place_object(self):
        if self.robot2_state == Robot2State.GRASPING:
            self.robot2_state = Robot2State.PLACING
            print("📦 Robot2 placing object")
            # TODO: 实现实际的放置逻辑

    def _update_robot2_pid_control(self):
        """更新Robot2的PID控制"""
        if self.robot2_state != Robot2State.MOVING or self.robot2_target_pos is None:
            return

        # 🎯 检查移动超时
        if self.current_step - self.robot2_movement_start_time > self.robot2_max_movement_time:
            print("⏰ Robot2 movement timeout, switching to IDLE")
            self.robot2_state = Robot2State.IDLE
            self.robot2_target_pos = None
            return

        # 🎯 获取当前状态
        pos = self.data.xpos[self.robot_2_rover_id][:2]
        quat = self.data.xquat[self.robot_2_rover_id]
        yaw = self.quaternion_to_yaw(quat)

        # 🎯 计算位置和朝向误差
        direction = self.robot2_target_pos - pos
        distance = np.linalg.norm(direction)
        
        # 🎯 根据距离动态选择目标朝向策略
        if distance > 0.5:
            target_heading = np.arctan2(direction[1], direction[0])
        else:
            target_heading = self.robot2_target_yaw
        
        yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 🎯 检查目标是否在车辆后方（添加倒车逻辑）
        car_forward = np.array([np.cos(yaw), np.sin(yaw)])
        if distance > 1e-6:
            target_direction = direction / distance
            forward_projection = np.dot(target_direction, car_forward)
            
            if forward_projection < -0.2:  # 目标在后方
                drive_sign = -1
                target_heading = (target_heading + np.pi) % (2 * np.pi)
                yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi
            else:
                drive_sign = 1
        else:
            drive_sign = 1

        # 🎯 PID控制计算
        params = self.robot2_pid_params
        drive_ctrl = drive_sign * (params['Kp_pos'] * distance + 
                                 params['Kd_pos'] * (distance - np.linalg.norm(self.robot2_prev_pos_error)))
        
        steer_ctrl = (params['Kp_yaw'] * yaw_error + 
                     params['Kd_yaw'] * (yaw_error - self.robot2_prev_yaw_error))
        
        # 🎯 倒车时转向反向
        if drive_sign == -1:
            steer_ctrl *= -1

        # 🎯 应用控制输出
        self.data.ctrl[self.robot2_drive_index] = np.clip(drive_ctrl, -3, 3)
        self.data.ctrl[self.robot2_steer_index] = np.clip(steer_ctrl, -0.9, 0.9)

        # 🎯 更新历史误差
        self.robot2_prev_pos_error = direction
        self.robot2_prev_yaw_error = yaw_error

        # 🎯 检查是否到达目标
        final_yaw_error = (self.robot2_target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        if distance < params['tol'] and abs(final_yaw_error) < params['tol_yaw']:
            self.robot2_state = Robot2State.IDLE
            self.robot2_target_pos = None
            self.data.ctrl[self.robot2_drive_index] = 0
            self.data.ctrl[self.robot2_steer_index] = 0
            print("✅ Robot2 reached target!")

    def _get_obs(self):
        """修改观察空间，添加Robot2状态信息"""
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]  
        robot2_quat = self.data.xquat[self.robot_2_rover_id]
        robot2_orientation = self.quaternion_to_yaw(robot2_quat)

        robot1_predicted_trajectory = self._predict_robot1_trajectory()
        collision_risks, min_distances = self._calculate_collision_risk_timeline(robot2_pos, robot1_predicted_trajectory)
        
        trajectory_features = self._extract_trajectory_features(robot1_predicted_trajectory, robot2_pos)

        target_pos = np.array(self.target_position_x_y) 
        target_rel = target_pos - robot2_pos  
        target_distance = np.linalg.norm(target_rel) 
        target_angle = np.arctan2(target_rel[1], target_rel[0]) 

        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]  
        robot1_rel = robot1_pos - robot2_pos  
        robot1_distance = np.linalg.norm(robot1_rel)  
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])  

        walls = {
            "left": -3.0, "right": 3.0, "front": 3.0, "back": -3.0    
        }
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],   
            walls["right"] - robot2_pos[0],  
            robot2_pos[1] - walls["back"],   
            walls["front"] - robot2_pos[1]   
        ])

        placing_place_1_pos = np.array([2.8, 1.0])  
        placing_1_rel = placing_place_1_pos - robot2_pos
        placing_1_distance = np.linalg.norm(placing_1_rel)

        placing_place_2_pos = np.array([2.8, -1.0])
        placing_2_rel = placing_place_2_pos - robot2_pos
        placing_2_distance = np.linalg.norm(placing_2_rel)

        # 🎯 Robot2状态信息（one-hot编码）
        robot2_state_onehot = np.zeros(len(Robot2State))
        robot2_state_onehot[self.robot2_state] = 1.0
        
        # 🎯 目标位置信息（如果有的话）
        if self.robot2_target_pos is not None:
            target_rel_current = self.robot2_target_pos - robot2_pos
            target_distance_current = np.linalg.norm(target_rel_current)
            target_angle_current = np.arctan2(target_rel_current[1], target_rel_current[0])
        else:
            target_distance_current = 0.0
            target_angle_current = 0.0
        
        max_position = 3.0     
        max_distance = 9     
        
        observation = np.concatenate([
            robot2_pos / max_position,                
            [robot2_orientation / np.pi],             # 🎯 添加朝向信息
            
            target_pos / max_position,               
            [target_distance / max_distance,           
            target_angle / np.pi],                   
            
            robot1_pos / max_position,               
            [robot1_distance / max_distance,          
            robot1_angle / np.pi],     

            trajectory_features,

            [np.max(collision_risks),                     # 最大风险
             np.mean(collision_risks),                    # 平均风险
             np.min(min_distances),                       # 最小距离
             np.argmax(collision_risks) / self.prediction_steps],  # 最危险时刻(归一化)
            
            wall_distances / max_distance,            

            placing_place_1_pos / max_position,       
            [placing_1_distance / max_distance],      
            
            placing_place_2_pos / max_position,       
            [placing_2_distance / max_distance],
            
            robot2_state_onehot,                     # 🎯 Robot2状态
            [target_distance_current / max_distance, # 🎯 当前目标距离
             target_angle_current / np.pi]           # 🎯 当前目标角度
        ], dtype=np.float32)

        return observation

    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def get_object_ids(self, model):
        object_ids = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("object") and name.endswith(":joint"):
                try:
                    num = int(name.split(":")[0][6:])
                    object_ids.append(num)
                except Exception:
                    continue
        return sorted(object_ids)

    def start_object_remover_threads(self, model, data, object_joint_ids):
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23

        threading.Thread(
            target=remove_object_on_plane,
            args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
            daemon=True
        ).start()

        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        threading.Thread(
            target=remove_object_on_plane,
            args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
            daemon=True
        ).start()

    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        threading.Thread(
            target=place_object_on_table,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()

    def check_robot_forbidden_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if ((body1_id in self.robot2_body_ids and geom2_name in self.forbidden_geoms) or
                (body2_id in self.robot2_body_ids and geom1_name in self.forbidden_geoms)):
                return True
        
        return False

    def check_robot_robot_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            is_robot1_involved = body1_id in self.robot1_body_ids or body2_id in self.robot1_body_ids
            is_robot2_involved = body1_id in self.robot2_body_ids or body2_id in self.robot2_body_ids
            
            if is_robot1_involved and is_robot2_involved:
                return True
        
        return False

    def reward_function(self, robot_2_rover_pos):
        robot2_quat = self.data.xquat[self.robot_2_rover_id]
        robot2_orientation = self.quaternion_to_yaw(robot2_quat)

        dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)

        progress_reward = 0
        if self.prev_dist is not None:
            progress_amount = (self.prev_dist - dist_to_target) * 100
            
            if dist_to_target > 4.0:
                coefficient = 2.0      
            elif dist_to_target > 3.0:
                coefficient = 3.0      
            elif dist_to_target > 2.0:
                coefficient = 4.0     
            elif dist_to_target > 1.0:
                coefficient = 5.0      
            elif dist_to_target > 0.5:
                coefficient = 6.0      
            elif dist_to_target > 0.2:
                coefficient = 8.0      
            else:
                coefficient = 10.0
            
            progress = progress_amount * coefficient
            
            if progress > 0:
                progress_reward = progress
            else:
                progress_reward = progress * 0.5

        robot_distance = self.get_robot_distance()
        safety_reward = self.calculate_safety_reward(robot_distance)

        time_penalty = -0.3

        # 🎯 状态奖励
        state_reward = 0
        if self.robot2_state == Robot2State.MOVING:
            state_reward = 1.0  # 鼓励移动
        elif self.robot2_state == Robot2State.IDLE:
            state_reward = -0.5  # 轻微惩罚空闲

        arrival_bonus = 0
        reached = (dist_to_target < 0.1)
        if reached:
            arrival_bonus = 200000

        total_reward = progress_reward + safety_reward + arrival_bonus + time_penalty + state_reward

        self.prev_dist = dist_to_target

        return total_reward, reached

    def get_robot_distance(self):
        robot1_pos = self.data.xpos[self.robot_1_rover_id]
        robot2_pos = self.data.xpos[self.robot_2_rover_id]
        return np.linalg.norm(robot1_pos - robot2_pos)

    def calculate_safety_reward(self, robot_distance):
        if robot_distance < 0.8:
            return 0
        elif robot_distance < 1.0:
            return 0.5
        else:
            return 1

    def quaternion_to_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def pid_controller(self, 
                        target_pos=np.zeros(2), 
                        Kp_pos=10.0, 
                        Kd_pos=1.0, 
                        Kp_yaw=3.0, 
                        Kd_yaw=0.1, 
                        max_steps=20000, tol=1e-3
                    ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0

        pos = self.data.xpos[self.robot_1_rover_id][:2]
        quat = self.data.xquat[self.robot_1_rover_id]
        yaw = self.quat_to_yaw(quat)

        direction = target_pos - pos
        distance = np.linalg.norm(direction)
        target_heading = np.arctan2(direction[1], direction[0])
        yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

        if abs(yaw_error) > np.pi / 2:
            drive_sign = -1
            
            yaw_error = ((target_heading + np.pi) % (2 * np.pi)) - yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        else:
            drive_sign = 1

        drive_ctrl = drive_sign * (Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error)))
        steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)

        self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -1.5, 1.5)
        self.data.ctrl[self.robot1_ghost_steer_index] = -1 * np.clip(steer_ctrl, -0.5, 0.5)

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        if distance < tol:
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0
            return True
        return False

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
    
    def _update_robot1_tracking(self):
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        self.robot1_recent_positions.append(robot1_pos.copy())
        
        # 只保留最近3帧位置
        if len(self.robot1_recent_positions) > 3:
            self.robot1_recent_positions.pop(0)

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
            
            # 碰撞风险计算
            if distance < 0.5:  # 危险距离
                risk = 1.0 - (distance / 0.5)
            elif distance < 1.0:  # 警告距离
                risk = 0.5 * (1.0 - (distance - 0.5) / 0.5)
            else:
                risk = 0.0
                
            collision_risks.append(risk)
        
        return np.array(collision_risks), np.array(min_distances)

    def _predict_robot2_trajectory(self, current_pos):
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