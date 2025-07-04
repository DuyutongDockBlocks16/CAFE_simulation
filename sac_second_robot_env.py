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

ACTION_SPACE_REDUCTION = 13  # Number of actuators to be reduced from the action space

class SacSecondRobotMuJoCoEnv(gym.Env):
    def __init__(self, xml_path, action_repeat=4):
        super().__init__()

        self.action_repeat = action_repeat
        if self.action_repeat > 1:
            print(f"🔄 Action Repeat enabled: {self.action_repeat}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        time_step = 0.005
        self.model.opt.timestep = time_step

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
        
        # self.target_area_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "placingplace1:low_plane")
        # self.target_position_x_y = [-1, -1.7] 
        # self.target_positions = [[0, 2], [-0.3, 1], [-0.5, 0]]
        self.target_positions = [[2, -2], [-2, -2]]
        self.target_position_x_y = random.choice(self.target_positions)

        # obs = self._get_obs()
        # print("Observation shape:", obs.shape)
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        # )
        num_actuators = self.model.nu

        self.low_bounds = np.array([-3.0, -0.9], dtype=np.float32)
        self.high_bounds = np.array([3.0, 0.9], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(num_actuators - ACTION_SPACE_REDUCTION,), 
            dtype=np.float32
        )

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
            "vacuum_sphere"         # vacuum gripper
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
            "robot2:link6"          # arm end effector
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
        self.prev_orientation = None
        self.target_orientation = -1.5708  # Target orientation in radians (90 degrees)
        self.init_dist = None
        self.static_counter = 0
        self.max_static_steps = 400 
        self._setup_observation_space()
        self.finished = True
        self.robot2_joint_indices = self._get_robot2_joint_indices()
        self.robot2_initial_qpos = {}
        self.robot2_initial_qvel = {}
        self.robot2_initial_ctrl = {}
        self._store_robot2_initial_states()

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
        
        # 🎯 获取robot2的执行器索引（基于您的XML actuator定义）
        robot2_actuator_names = [
            "robot2:Joint1",              # 机械臂执行器
            "robot2:Joint2",
            "robot2:Joint3", 
            "robot2:Joint4",
            "robot2:Joint5",
            "robot2:drive",               # 后轮差速驱动
            "robot2:ghost-steer"          # 转向控制
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

            self.first_robot_controller.set_state(FiniteState.IDLE)
            self.first_robot_controller.reset_all_joints()

            mujoco.mj_forward(self.model, self.data)
            self.finished = False

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
        """修改step函数以适配HER"""
        
        normalized_action = np.clip(action, -1, 1)
        real_action = self.low_bounds + (normalized_action + 1) * (self.high_bounds - self.low_bounds) / 2

        terminated = False
        truncated = False
        self.first_robot_controller.step(self.shared_state["current_object_position"])
        self.data.ctrl[ACTION_SPACE_REDUCTION:ACTION_SPACE_REDUCTION+len(real_action)] = real_action
        status = self.first_robot_controller.get_status()
        
        if self.shared_state["current_object_index"] >= len(self.object_joint_ids) and status == FiniteState.IDLE:
            print("All objects have been placed. Exit")
            terminated = True
            self.finished = True

        mujoco.mj_step(self.model, self.data)

        # 🎯 获取Dict格式观察
        obs = self._get_obs()
        
        # 🎯 使用HER的奖励函数
        reward = self.compute_reward(
            obs["achieved_goal"], 
            obs["desired_goal"], 
            {}
        )
        
        # 🎯 检查是否成功（用于HER的success信息）
        distance_to_target = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        success = distance_to_target < 0.2
        
        # 🎯 碰撞和其他检查
        
        
        if self.check_robot_forbidden_collision():
            print("Robot collision with forbidden area detected! Terminating episode.")
            reward -= 10.0  # HER通常使用较小的惩罚
            terminated = True

        if self.check_robot_robot_collision():
            print("Robot-robot collision detected! Terminating episode.")
            reward -= 10.0
            terminated = True

        if not np.all(np.isfinite(self.data.qacc)) or np.any(np.abs(self.data.qacc) > 1e7):
            print("QACC error detected! Simulation unstable, exiting loop.")
            truncated = True
            self.finished = True

        if success:
            print("Robot2 has reached the target area! Success!")
            terminated = True

        self.current_step += 1

        if self.current_step >= self.max_steps:
            terminated = True
        
        # 🎯 HER需要的info信息
        info = {
            "is_success": success,
            "distance_to_target": distance_to_target
        }
        
        return obs, reward, terminated, truncated, info

    def _setup_observation_space(self):

        obs_dim = 15 
        
        obs_low = np.array([
            -1.0, -1.0,     # robot2_pos (归一化后)
            -2.0, -2.0,     # robot1_rel (可能超出归一化范围)
            0.0,            # robot1_distance
            -1.0, -1.0,     # robot1_angle_sin/cos
            0.0, 0.0, 0.0, 0.0,  # wall_distances
            -2.0, -2.0,     # nearest_place_rel
            0.0,            # nearest_place_distance
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0, 1.0,       # robot2_pos
            2.0, 2.0,       # robot1_rel
            1.0,            # robot1_distance
            1.0, 1.0,       # robot1_angle_sin/cos
            1.0, 1.0, 1.0, 1.0,  # wall_distances
            2.0, 2.0,       # nearest_place_rel
            1.0,            # nearest_place_distance
        ], dtype=np.float32)
        
        observation_space = gym.spaces.Box(
            low=obs_low, 
            high=obs_high, 
            dtype=np.float32
        )
        
        # 🎯 目标空间（2D位置坐标）
        goal_space = gym.spaces.Box(
            low=np.array([-3.0, -3.0], dtype=np.float32),
            high=np.array([3.0, 3.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 🎯 组合为Dict观察空间
        self.observation_space = gym.spaces.Dict({
            "observation": observation_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space
        })

    def _get_obs(self):
        """HER兼容的Dict格式观察空间"""
        
        robot2_pos = self.data.xpos[self.robot_2_rover_id][:2]
        target_pos = np.array(self.target_position_x_y)
        robot1_pos = self.data.xpos[self.robot_1_rover_id][:2]
        
        # 🎯 计算环境观察（不包含目标相关信息）
        robot1_rel = robot1_pos - robot2_pos
        robot1_distance = np.linalg.norm(robot1_rel)
        robot1_angle = np.arctan2(robot1_rel[1], robot1_rel[0])
        robot1_angle_sin = np.sin(robot1_angle)
        robot1_angle_cos = np.cos(robot1_angle)
        
        # 墙壁距离
        walls = {"left": -3.0, "right": 3.0, "front": 3.0, "back": -3.0}
        wall_distances = np.array([
            robot2_pos[0] - walls["left"],
            walls["right"] - robot2_pos[0],
            robot2_pos[1] - walls["back"],
            walls["front"] - robot2_pos[1]
        ])
        
        # 放置点信息
        placing_place_1_pos = np.array([2.8, 1.0])
        placing_place_2_pos = np.array([2.8, -1.0])
        
        dist1 = np.linalg.norm(placing_place_1_pos - robot2_pos)
        dist2 = np.linalg.norm(placing_place_2_pos - robot2_pos)
        
        if dist1 < dist2:
            nearest_place_distance = dist1
            nearest_place_rel = placing_place_1_pos - robot2_pos
        else:
            nearest_place_distance = dist2
            nearest_place_rel = placing_place_2_pos - robot2_pos
        

        max_position = 3.0
        max_distance = 8.5

        observation = np.concatenate([
            robot2_pos / max_position,                    # 机器人2位置 (2维)
            robot1_rel / max_position,                    # 机器人1相对位置 (2维)
            [robot1_distance / max_distance],             # 机器人1距离 (1维)
            [robot1_angle_sin, robot1_angle_cos],         # 机器人1角度 (2维)
            wall_distances / max_distance,                # 墙壁距离 (4维)
            nearest_place_rel / max_position,             # 最近放置点相对位置 (2维)
            [nearest_place_distance / max_distance],      # 最近放置点距离 (1维)
        ], dtype=np.float32)
        
        obs_dict = {
            "observation": observation,                    # 环境状态（15维）
            "achieved_goal": robot2_pos.astype(np.float32),  # 当前位置作为已达成目标
            "desired_goal": target_pos.astype(np.float32)    # 目标位置作为期望目标
        }
        
        return obs_dict

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        success_threshold = 0.2

        reward = -(distance > success_threshold).astype(np.float32)
        
        return reward

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
                # extract the N from the name
                try:
                    num = int(name.split(":")[0][6:])  # "objectN:joint" -> N
                    object_ids.append(num)
                except Exception:
                    continue
        return sorted(object_ids)

    def start_object_remover_threads(self, model, data, object_joint_ids):
        # lower plane parameters
        lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        lower_plane_radius = 0.23
        lower_plane_z = 0.23

        # threading.Thread(
        #     target=remove_object_on_plane,
        #     args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
        #     daemon=True
        # ).start()
        threading.Thread(
            target=remove_object_on_plane_with_step_counter,
            args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
            daemon=True
        ).start()

        # upper plane parameters
        upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
        upper_plane_radius = 0.15
        upper_plane_z = 0.43

        # threading.Thread(
        #     target=remove_object_on_plane,
        #     args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
        #     daemon=True
        # ).start()

        threading.Thread(
            target=remove_object_on_plane_with_step_counter,
            args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
            daemon=True
        ).start()

    def start_object_placer_thread(self, model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
        # object positions parameters
        threading.Thread(
            target=place_object_on_table,
            args=(model, data, left_object_position, right_object_position, object_joint_ids),
            kwargs={"shared_state": shared_state},
            daemon=True
        ).start()

    def check_robot_forbidden_collision(self):
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

    def check_robot_distance_violation(self):
        """Check if the distance between two robot chassis is less than the safe distance"""
        
        # Get positions of both robot chassis
        robot1_pos = self.data.xpos[self.robot1_rover_id]
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(robot1_pos - robot2_pos)
        
        return distance < self.safe_distance

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
                progress_reward = progress * 1.0


        robot_distance = self.get_robot_distance()
        safety_reward = self.calculate_safety_reward(robot_distance)

        time_penalty = -0.3

        arrival_bonus = 0

        reached = (dist_to_target < 0.2)
        if reached:
            arrival_bonus = 30000

        total_reward = progress_reward + safety_reward + arrival_bonus + time_penalty

        self.prev_dist = dist_to_target

        return total_reward, reached

    def simple_reward_function(self, robot_2_rover_pos):
        dist_to_target = np.linalg.norm(robot_2_rover_pos - self.target_position_x_y)
        reward = -(abs(robot_2_rover_pos[0] - self.target_position_x_y[0]) + abs(robot_2_rover_pos[1] - self.target_position_x_y[1])) * 0.1

        reached = (dist_to_target < 0.2)

        return reward, reached

    def get_robot_distance(self):
        """Calculate the distance between the two robots"""
        robot1_pos = self.data.xpos[self.robot1_rover_id]
        robot2_pos = self.data.xpos[self.robot2_rover_id]
        return np.linalg.norm(robot1_pos - robot2_pos)

    def calculate_safety_reward(self, robot_distance):
        """Safety distance reward design"""
        if robot_distance < 0.8:     # Collision
            return 0            # Large penalty
        elif robot_distance < 1.0:   # Danger zone
            return 0.2             # Medium penalty  
        # elif robot_distance < 2.0: 
        #     return 1               
        else:                        # Safe zone
            return 0.3

    def check_robot_robot_collision(self):
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
                return True
        
        return False

    def calculate_static_penalty(self, action, current_pos):
        penalty = 0
        

        if self.prev_position is not None:
            position_change = np.linalg.norm(current_pos - self.prev_position)
            
            if position_change < 0.05:  
                self.static_counter += 1
                if self.static_counter > 50:  
                    
                    penalty -= min((self.static_counter - 50) * 0.05, -5) 
            else:
                self.static_counter = max(0, self.static_counter - 3)  
                penalty += 0.05 
        
        self.prev_position = current_pos.copy()
        
        return penalty

    def quaternion_to_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def calculate_orientation_progress_reward(self, current_orientation):

        if not hasattr(self, 'prev_orientation') or self.prev_orientation is None:
            self.prev_orientation = current_orientation
            return 0.0
        
        current_error = abs(current_orientation - self.target_orientation)
        prev_error = abs(self.prev_orientation - self.target_orientation)


        if current_error > np.pi:
            current_error = 2 * np.pi - current_error
        if prev_error > np.pi:
            prev_error = 2 * np.pi - prev_error
        
        orientation_progress = (prev_error - current_error) * 10  
        
        self.prev_orientation = current_orientation
        
        return orientation_progress

    def calculate_orientation_reward(self, current_orientation, distance_to_target):

        orientation_reward = 0.0   
        angle_error = abs(current_orientation - self.target_orientation)

        if angle_error > np.pi:
            angle_error = 2 * np.pi - angle_error
        
        if distance_to_target < 0.5:
            if angle_error < 0.05:  
                orientation_reward = 10.0  
            elif angle_error < 0.1:  
                orientation_reward = 5.0   
            else:  
                orientation_reward = 2.0   

        elif distance_to_target < 1.0:
            if angle_error < 0.1:
                orientation_reward = 3.0
            elif angle_error < 0.2:
                orientation_reward = 1.0
            else:
                orientation_reward = 0.0

        elif distance_to_target < 2.0:
            if angle_error < 0.2:
                orientation_reward = 1.0
            else:
                orientation_reward = 0.0

        elif distance_to_target < 2.0:
            if angle_error < 0.2:
                orientation_reward = 1.0
            else:
                orientation_reward = 0.0

        else:
            if angle_error < 0.5:
                orientation_reward = 0.5
            else:
                orientation_reward = 0.0
        
        return orientation_reward
