import gymnasium as gym
import sec_robot_env 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from time import sleep
import mujoco.viewer
import time
import os
from datetime import datetime
from config.env_config import Direction, Layer, FiniteState
from config.training_config import APPROACHING_MODEL_NAME, SUCCESS_THRESHOLD
from callbacks.episode_data_collector import EpisodeBatchCollector
from callbacks.success_check_point_saver import SuccessCheckpointCallback
from callbacks.training_renderer import RenderCallback
from callbacks.ent_coefficient_scheduler import EntCoefficientScheduler
from callbacks.learning_rate_scheduler import LearningRateScheduler
from utils.mujoco_state_saver import save_mujoco_state_to_file
from utils.mujoco_state_loader import load_mujoco_state_from_file, restore_mujoco_state, view_saved_state
import numpy as np
import pickle
import json

gym.register(
    id="SecondRobotMuJoCoEnv-v0",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={
        "xml_path": "xml/scene_mirobot.xml",
    }
)

class HybridController:
    def __init__(self, model, data, 
            navigation_model_paths=[],
            picking_model_paths=[],
            placing_model_paths=[]
        ):
        self.left_object_position = [1, -2.5, 0.28]
        self.right_object_position = [-1, -2.5, 0.28]

        self.model, self.data = self._get_data_and_model()
        
        self.first_robot_controller = MirobotController(self.model, self.data, left_pos, right_pos)
        
        object_ids = self._get_object_ids(self.model)
        
        self.object_joint_ids = []
        
        for i in object_ids:
            joint_name = f"object{i}:joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.object_joint_ids.append((i, joint_id))
        
        self.shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}

        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_2_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot2:rover")

        self.picking_positions = [
            [1, -2.45], # left
            [-1, -2.45],  # right
        ]

        self.placing_positions = [
        ]  

        self.target_position_x_y = None

        self.robot1_recent_positions = []
        self.robot2_recent_positions = []
        self.prediction_steps = 5

        self.current_step = 0
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_ctrl = np.copy(self.data.ctrl)
        
        self.max_position = 3.0     
        self.max_speed = 2.0        
        self.max_distance = 8.0

    def _get_data_and_model(self):
        model = mujoco.MjModel.from_xml_path("xml/scene_mirobot.xml")
        data = mujoco.MjData(model)
        time_step = 0.005
        model.opt.timestep = time_step  
        return model, data

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
    
    def get_robot_observation(self):
        robot_qpos = self.mujoco_data.qpos[:6]  
        robot_qvel = self.mujoco_data.qvel[:6] 
        
        obs = np.concatenate([
            robot_qpos,
            robot_qvel,
            # target_pos,
            # end_effector_pos,
            # 其他需要的观察...
        ]).astype(np.float32)
        
        return obs
    
    def step(self, target_position, use_ppo=None):
        """
        执行一步控制
        """
        if use_ppo is None:
            use_ppo = self.use_ppo
        
        if use_ppo and self.ppo_model is not None:
            obs = self.get_robot_observation()
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            
            self.apply_ppo_action(action)
            
        else:
            # 🎯 使用原有的手动控制器
            self.manual_controller.step(target_position)

    def _get_navigation_obs(self):
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
            robot2_pos / self.max_position,                    # [2] 绝对位置
            robot2_vel / self.max_speed,                       # [2] 速度 
            [robot2_orientation / np.pi],                 # [1] 朝向 

            # 🎯 目标相对信息（增强版）
            target_rel_normalized,                        # [2] 目标相对位置向量 
            [target_distance / self.max_distance],             # [1] 目标距离
            [target_angle / np.pi],                       # [1] 目标世界角度
            [target_relative_angle / np.pi],              # [1] 目标相对角度 

            # 🎯 机器人1相对信息（增强版）
            robot1_rel_normalized,                        # [2] 机器人1相对位置 
            [robot1_distance / self.max_distance],             # [1] 机器人1距离
            [robot1_relative_angle / np.pi],              # [1] 机器人1相对角度 

            trajectory_features,

            [np.max(collision_risks),                     # 最大风险
             np.mean(collision_risks),                    # 平均风险
             np.min(min_distances),                       # 最小距离
             np.argmax(collision_risks) / self.prediction_steps],  # 最危险时刻(归一化)
            
            # 🎯 环境相对信息
            wall_distances / self.max_distance,                # [4] 墙壁距离

            # 🎯 放置点相对信息（增强版）
            placing_1_rel_normalized,                     # [2] 放置点1相对位置 
            [placing_1_distance / self.max_distance],          # [1] 放置点1距离
            [placing_1_angle / np.pi],                    # [1] 放置点1相对角度 
            
            placing_2_rel_normalized,                     # [2] 放置点2相对位置 
            [placing_2_distance / self.max_distance],          # [1] 放置点2距离
            [placing_2_angle / np.pi],                    # [1] 放置点2相对角度 
        ], dtype=np.float32)

        return observation
    
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


    def apply_ppo_action(self, action):
        """
        将PPO的动作应用到MuJoCo
        """
        # 🎯 根据你的动作空间调整
        # 假设action是关节控制命令
        self.mujoco_data.ctrl[:len(action)] = action
    
    def get_status(self):
        return self.manual_controller.get_status()
    
    def switch_to_ppo(self):
        """切换到PPO控制"""
        if self.ppo_model is not None:
            self.use_ppo = True
            print("🤖 Switched to PPO control")
    
    def switch_to_manual(self):
        """切换到手动控制"""
        self.use_ppo = False
        print("🎮 Switched to manual control")

def approach_model_implementation(env):
    ppo_model = PPO.load(APPROACHING_MODEL_NAME, env=env)
    print(f"✅ Loaded PPO model: {APPROACHING_MODEL_NAME}")
    
    obs, info = env.reset()
    print("🎬 Starting model demonstration...")
    
    env.render()
    sleep(15) 
    
    episode_count = 0
    max_episodes = 1 
    
    while episode_count < max_episodes:
        print(f"\n🎯 Episode {episode_count + 1}/{max_episodes}")
        
        step_count = 0
        max_steps_per_episode = 10000  
        
        while step_count < max_steps_per_episode:
            env.render()

            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            
            if step_count % 500 == 0:
                print(f"   Step {step_count}, Reward: {reward:.2f}")
            
            if terminated or truncated:
                print(f"   Episode ended after {step_count} steps")
                print(f"   Final reward: {reward:.2f}")
                print(f"   Terminated: {terminated}, Truncated: {truncated}")
                # env.unwrapped.data.ctrl[:] = 0
                # mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)
                save_mujoco_state_to_file(env.unwrapped.model, env.unwrapped.data)

                # obs, info = env.reset()
                break
        
        episode_count += 1
        
        if episode_count < max_episodes:
            print("   🔄 Starting next episode in 3 seconds...")
            sleep(3)
    
    print("🎬 Demonstration completed!")
    
    mujoco_model = env.unwrapped.model
    mujoco_data = env.unwrapped.data
    
    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()
    env.close()
    
    print("🔍 Launching passive viewer...")
    sleep(5)
    
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        print("🎮 Passive viewer launched!")
        print("   - Press ESC to exit viewer")
        print("   - Use mouse to rotate view")
        print("   - Use scroll to zoom")
        
        last_time = time.time()
        frame_count = 0
        
        while viewer.is_running():
            mujoco.mj_step(mujoco_model, mujoco_data)
            viewer.sync()
            
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_time >= 5.0: 
                fps = frame_count / (current_time - last_time)
                print(f"📊 Simulation FPS: {fps:.1f}")
                frame_count = 0
                last_time = current_time
    
    print("✅ Viewer closed. Implementation demo finished!")

def setup_initial_state(model, data):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

def get_navigation_observation_from_mujoco(model, data):
    """
    从MuJoCo数据中提取观察值
    这个函数需要复制你原来env中的观察逻辑
    """
    # 🎯 获取机器人状态
    robot_pos = data.qpos[:3]  # 位置
    robot_vel = data.qvel[:3]  # 速度
    
    # 🎯 获取目标位置 (需要根据你的具体设置调整)
    target_pos = np.array([2.0, 2.0])  # 硬编码或从模型中获取
    
    # 🎯 计算相对位置
    relative_pos = target_pos - robot_pos[:2]
    
    # 🎯 构建观察向量 (复制你原来env的obs逻辑)
    obs = np.concatenate([
        robot_pos,      # 机器人位置
        robot_vel,      # 机器人速度
        relative_pos,   # 到目标的相对位置
        # 添加其他你需要的观察
    ]).astype(np.float32)
    
    return obs

if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    approach_model_implementation(approach_env)

    # state_file = "saved_states/robot_state_20250708_151607.pkl"
    # view_saved_state(state_file)