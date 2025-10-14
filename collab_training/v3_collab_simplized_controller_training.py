import os
import torch

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.insert(0, parent_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_device('cpu')

print("🖥️ 强制使用 CPU 进行训练")
print(f"PyTorch device: {torch.device('cpu')}")
print(f"CUDA available: {torch.cuda.is_available()}")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from time import sleep
import mujoco.viewer
import time
import os
from datetime import datetime
from config.training_config import COLLAB_2_MODEL_NAME, DRIVER_MODEL_NAME, FIXED_MODEL_NAME, V1_MODEL_NAME
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
import torch
# from sb3_contrib import PPO
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.wrappers import ActionMasker

gym.register(
    id="V3CollabHybridMuJoCoEnv-v0",
    entry_point="v3_collab_hybrid_robot_env:V3CollabHybridMuJoCoEnv",
    # kwargs={
    #     "xml_path": "xml/scene_mirobot.xml",
    #     "state_filepath": "saved_states/robot_state_20250723_093442.pkl"
    # }
)

def make_env(rank, seed=0):
    """Factory function to create environment"""
    def _init():
        env = gym.make(
            "V3CollabHybridMuJoCoEnv-v0"
            )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def driver_model_training(env, load_model_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/driver_episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    combined_callback = CallbackList([
        RenderCallback(env),
        # SuccessCheckpointCallback("./checkpoints"),
        episode_collector
    ])
    
    if load_model_path is not None:
        if not os.path.exists(load_model_path):
            print(f"❌ Model {load_model_path} not found!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}_{os.path.basename(load_model_path)}"
        os.system(f"cp {load_model_path} {backup_name}")
        print(f"📁 Created backup: {backup_name}")

        model = PPO.load(load_model_path, env=env)
        print(f"✅ Successfully loaded model from: {load_model_path}")
        
        model.tensorboard_log = f"./ppo_logs/continued_{timestamp}/"
        
        import re
        match = re.search(r'(\d+)K', load_model_path)
        if match:
            loaded_steps = int(match.group(1)) * 1000
            print(f"   Continuing from approximately {loaded_steps:,} steps")
        else:
            loaded_steps = 0
            print("   Could not determine previous training steps from filename")
            
    else:
        print("🆕 Creating new PPO model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4,     # Learning rate
            n_steps=8192,           # Collect 8192 steps of experience each time
            batch_size=256,          # Process 256 samples per batch
            n_epochs=10,            
            ent_coef=0.02,          
            clip_range=0.2,          
            gae_lambda=0.95,         
            vf_coef=0.5,            
            tensorboard_log="./ppo_logs/")  # Log save path
        loaded_steps = 0

    save_interval = 500_000 
    # total_additional_steps = 1_600_000_000
    total_additional_steps = 80_000_000
    
    print(f"🚀 Starting training...")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Save interval: {save_interval:,} steps")
    print(f"   Model type: {'Continued' if load_model_path else 'New'}")
    
    num_iterations = total_additional_steps // save_interval
    
    for i in range(num_iterations):
        print(f"\n--- Training Progress: {i+1}/{num_iterations} ---")
        
        model.learn(total_timesteps=save_interval,      
                   callback=combined_callback, 
                   reset_num_timesteps=False)          # Don't reset timestep counter
        
        current_total_steps = loaded_steps + (i + 1) * save_interval
        
        if load_model_path:
            model_name = f"driver_model_continued_{current_total_steps // 1000}K"
        else:
            model_name = f"driver_model_car_{current_total_steps // 1000}K"
            
        model.save(model_name)
        print(f"💾 Saved: {model_name}.zip ({current_total_steps:,} total steps)")
        
        if (i + 1) * save_interval % 1_000_000 == 0:
            millions = current_total_steps // 1_000_000
            print(f"🎉 Milestone: Reached {millions}M total steps!")
    
    final_total_steps = loaded_steps + total_additional_steps
    if load_model_path:
        final_model_name = f"driver_model_continued_{final_total_steps // 1000}K_final"
    else:
        final_model_name = f"driver_model_car_{final_total_steps // 1000}K_final"
    
    model.save(final_model_name)
    
    print(f"\n🎊 ============ TRAINING COMPLETED! ============")
    print(f"📊 Training Summary:")
    if load_model_path:
        print(f"   Original model: {load_model_path}")
        print(f"   Starting steps: {loaded_steps:,}")
    else:
        print(f"   Training type: New model from scratch")
        print(f"   Starting steps: 0")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Final total steps: {final_total_steps:,}")
    print(f"   Final model: {final_model_name}.zip")
    
    env.close()
    
def driver_model_training_parallel(load_model_path=None, num_envs=14):
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../logs/driver_episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    if load_model_path is not None:
        if not os.path.exists(load_model_path):
            print(f"❌ Model {load_model_path} not found!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}_{os.path.basename(load_model_path)}"
        os.system(f"cp {load_model_path} {backup_name}")
        print(f"📁 Created backup: {backup_name}")

        model = PPO.load(load_model_path, env=env, device='cpu',)

        model.ent_coef = 0.1
        model.learning_rate = 3e-4

        print(f"✅ Successfully loaded model from: {load_model_path}")
        print(f"🔄 Using {num_envs} parallel environments")
        
        model.tensorboard_log = f"./ppo_logs/driver_model_continued_parallel_{timestamp}/"
        
        import re
        match = re.search(r'(\d+)K', load_model_path)
        if match:
            loaded_steps = int(match.group(1)) * 1000
            print(f"   Continuing from approximately {loaded_steps:,} steps")
        else:
            loaded_steps = 0
            
    else:
        print("🆕 Creating new PPO model with parallel environments...")
        print(f"🔄 Using {num_envs} parallel environments")

        model = PPO(
            "MlpPolicy",
            env, verbose=1,
            device='cpu',
                    learning_rate=3e-4,     
                    n_steps=2048,           # 调整为并行环境合适的值
                    batch_size=512,          
                    n_epochs=8,            
                    ent_coef=0.02,          
                    clip_range=0.15,          
                    gae_lambda=0.95,         
                    vf_coef=1.0,            
                    # max_grad_norm=0.3,
                    tensorboard_log="./ppo_logs/")
        loaded_steps = 0

    # total_additional_steps = 160_000_000
    # total_additional_steps = 500_000
    total_additional_steps = 12_000_000

    ent_scheduler = EntCoefficientScheduler(
        initial_ent_coef=0.02,  
        final_ent_coef=0.02,  
        # initial_ent_coef=0.1,  
        # final_ent_coef=0.1,           
        # final_ent_coef=0.0005,          
        total_timesteps=total_additional_steps,
        schedule_type='exponential',    
        verbose=1
    )
    
    lr_scheduler = LearningRateScheduler(
        initial_lr=5e-5,
        final_lr=1e-5,
        total_timesteps=total_additional_steps,
        schedule_type='linear',
        verbose=1
    )

    combined_callback = CallbackList([
        # SuccessCheckpointCallback("./checkpoints"),
        episode_collector,
        ent_scheduler,
    ])
    
    print(f"🚀 Starting optimized parallel training...")
    print(f"   Parallel environments: {num_envs}")
    print(f"   Total additional steps: {total_additional_steps:,}")
    
    try:
        print(f"\n🎯 Starting continuous training for {total_additional_steps:,} steps...")
        
        model.learn(
            total_timesteps=total_additional_steps,      
            callback=combined_callback, 
            reset_num_timesteps=False
        )
        
        print(f"\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n Training interrupted by user")
    except Exception as e:
        print(f"\n Training error: {e}")
        import traceback
        traceback.print_exc()
    
    final_total_steps = loaded_steps + total_additional_steps
    
    if load_model_path:
        final_model_name = f"final_driver_model_continued_{final_total_steps // 1000}K_{timestamp}"
    else:
        final_model_name = f"final_driver_model_{final_total_steps // 1000}K_{timestamp}"

    try:
        model.save(final_model_name)
        print(f"\n💾 ============ MODEL SAVED ============")
        print(f"📁 Final model: {final_model_name}.zip")
        print(f"📊 Total training steps: {final_total_steps:,}")
        print(f"🕒 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📈 Final training parameters:")
        print(f"   Learning rate: {model.learning_rate:.2e}")
        print(f"   Entropy coefficient: {model.ent_coef:.6f}")
        
    except Exception as e:
        print(f"❌ Error saving final model: {e}")
    
    print(f"\n🎊 ============ TRAINING COMPLETED! ============")
    print(f"📊 Training Summary:")
    print(f"   Parallel environments: {num_envs}")
    if load_model_path:
        print(f"   Original model: {load_model_path}")
        print(f"   Starting steps: {loaded_steps:,}")
    else:
        print(f"   Training type: New model from scratch")
        print(f"   Starting steps: 0")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Final total steps: {final_total_steps:,}")
    print(f"   Models saved: 1 (final only)")
    
    env.close()
    
# def driver_model_test_single_episode(env):
#     """测试单个episode的简化版本"""
    
#     def mask_fn(env):
#         return env.get_action_mask()
    
#     env = ActionMasker(env, mask_fn)
#     model = PPO.load(DRIVER_MODEL_NAME, env=env)
    
#     obs, info = env.reset()
#     env.render()
#     sleep(3)
    
#     step_count = 0
#     print("🚀 开始单episode测试...")
    
#     while True:
#         env.render()
        
#         # 获取状态信息
#         fsm_state = env.unwrapped.second_robot_status
#         action_mask = env.unwrapped.get_action_mask()
#         valid_actions = np.where(action_mask)[0]
        
#         # 每50步打印一次状态
#         if step_count % 50 == 0:
#             print(f"\nStep {step_count}:")
#             print(f"  FSM State: {fsm_state}")
#             print(f"  Valid actions: {valid_actions.tolist()}")
        
#         # 预测并执行动作
#         action, _ = model.predict(obs, deterministic=True)
#         print(f"  Action: {action}", end="")
        
#         obs, reward, terminated, truncated, info = env.step(action)
#         step_count += 1
        
#         if step_count % 50 == 0:
#             print(f", Reward: {reward:.3f}")
#         else:
#             print()
        
#         if terminated or truncated:
#             print(f"\n🏁 Episode结束!")
#             print(f"  总步数: {step_count}")
#             print(f"  终止原因: {'成功完成' if terminated else '超时截断'}")
#             print(f"  最终奖励: {reward:.3f}")
#             break
        
#         if step_count > 10000:  # 防止无限循环
#             print("⚠️ 达到最大步数限制")
#             break
        
#         sleep(0.01)
    
#     env.close()
#     print("✅ 测试完成")

def driver_model_implementation(env):
    model = PPO.load(COLLAB_2_MODEL_NAME, env=env)
    obs, info = env.reset()

    env.render()
    sleep(10)

    for _ in range(200000000000):
        env.render()  # Render at every step
        sleep(0.01)
        # action, _ = model.predict(obs, deterministic=True)
        action, _ = model.predict(obs, deterministic=False)
        # save all action to file
        # with open("picking_obs_log.txt", "a") as f:
        #     f.write(f"{obs.tolist()}\n")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # obs, info = env.reset()
            # env.unwrapped.data.ctrl[:] = 0
            mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  
            break

    # model = env.unwrapped.model
    # data = env.unwrapped.data

    env.close()

    # sleep(20)

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     print("Press ESC to exit viewer...")
    #     last_time = time.time()
    #     frame_count = 0
    #     while viewer.is_running():
    #         mujoco.mj_step(model, data)
    #         viewer.sync()
    #         frame_count += 1
    #         now = time.time()
    #         if now - last_time >= 1.0:
    #             # print(f"Simulated FPS: {frame_count}")
    #             frame_count = 0
    #             last_time = now
    
def data_collection(env):
    # python v3_collab_simplized_controller_training.py 2>&1 | tee data_collection_$(date +%Y%m%d_%H%M%S).log
    model = PPO.load(COLLAB_2_MODEL_NAME, env=env)
    
    range_number = 100
    
    for i in range(range_number):
        print(f"\n=== Data Collection Episode {i+1}/{range_number} ===")
        obs, info = env.reset()

        # env.render()
        # sleep(10)
        
        episode_start_time = env.unwrapped.data.time

        for _ in range(200000000000):
            # print(f"\r  Sim Time: {env.unwrapped.data.time:.2f}s", end="")
            # env.render() 
            # sleep(0.01)
            action, _ = model.predict(obs, deterministic=False)
            # print(f" Action: {action}", end="")
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                env.unwrapped.data.ctrl[:] = 0
                mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  
                mujoco_timestep = env.unwrapped.model.opt.timestep
                current_simulation_time = env.unwrapped.data.time
                episode_simulation_time = current_simulation_time - episode_start_time
                total_mujoco_steps = int(episode_simulation_time / mujoco_timestep)
                
                print(f"  - Total MuJoCo steps executed: {total_mujoco_steps}")
                break
        
        # sleep(10)
        # env.close()
    
    env.close()
    
def driver_model_training_episode_save(env, load_model_path=None, save_every_episodes=1, enable_backup=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../logs/driver_episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=1,
        verbose=1
    )
    
    combined_callback = CallbackList([
        episode_collector,
    ])
    
    if load_model_path is not None:
        model = PPO.load(load_model_path, env=env)
        print(f"✅ Loaded model from: {load_model_path}")
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            tensorboard_log="./ppo_logs/"
        )
    
    episode_count = 0
    total_episodes = 8000
    
    for episode in range(total_episodes):
        print(f"\n=== Episode {episode+1}/{total_episodes} ===")
        
        # 训练一个episode
        obs, info = env.reset()
        done = False
        steps_in_episode = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps_in_episode += 1
            
            if steps_in_episode > 8000: 
                print("⚠️ Reached maximum steps, breaking...")
                done = True
                break
            
        if steps_in_episode > 0:
            model.learn(
                total_timesteps=steps_in_episode, 
                reset_num_timesteps=False, 
                callback=combined_callback
            )
        
        episode_count += 1

        # 每隔指定episode保存模型
        if episode_count % save_every_episodes == 0:
            if enable_backup:
                # 备份模式：如果存在旧模型，先备份
                if os.path.exists(f"{FIXED_MODEL_NAME}"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"backup_ep{episode_count-save_every_episodes}_{timestamp}.zip"
                    os.rename(f"{FIXED_MODEL_NAME}", f"{backup_name}")
                    print(f"📁 Backed up previous model to: {backup_name}")
            else:
                # 无备份模式：直接覆盖
                if os.path.exists(f"{FIXED_MODEL_NAME}"):
                    print(f"🔄 Overwriting existing model: {FIXED_MODEL_NAME}")
            
            # 保存新模型到固定名字
            model.save(FIXED_MODEL_NAME)
            print(f"💾 Saved model after episode {episode_count}: {FIXED_MODEL_NAME}")
    
    env.close()
    
def driver_model_training_timestep_based_parallel_safe(load_model_path=None, num_envs=14):
    
    # 创建并行环境（在同一个进程内）
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../logs/driver_episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    combined_callback = CallbackList([
        episode_collector,
    ])
    
    if load_model_path:
        model = PPO.load(load_model_path, env=env, device='cpu')
        print(f"✅ Loaded initial model: {load_model_path}")
    else:
        model = PPO("MlpPolicy", env, verbose=1, device='cpu', n_steps=2048)
    
    for round in range(8):  # 减少轮数，因为并行效率高
        print(f"\n=== Round {round+1}/8 ===")

        # 每轮重新加载模型（如果需要逐步改进）
        if round > 0:
            model = PPO.load(f"{FIXED_MODEL_NAME}", env=env, device='cpu')
            print(f"🔄 Reloaded model for round {round+1}")
        
        model.learn(
            total_timesteps=4_000_000, 
            reset_num_timesteps=False,
            callback=combined_callback
        )
         
        # 保存模型
        model.save(FIXED_MODEL_NAME)
        print(f"💾 Round {round+1}: Model improved and saved")
    
    env.close()

    
if __name__ == "__main__":
    driver_env = gym.make("V3CollabHybridMuJoCoEnv-v0")
    # driver_model_training(driver_env)
    # driver_model_training(driver_env, load_model_path=COLLAB_2_MODEL_NAME)
    # driver_model_training_parallel(load_model_path=COLLAB_2_MODEL_NAME, num_envs=14)
    # driver_model_training_parallel(load_model_path=COLLAB_2_MODEL_NAME, num_envs=8)
    # driver_model_test_single_episode(driver_env)
    # driver_model_implementation(driver_env)
    data_collection(driver_env)
    
    # driver_model_training_timestep_based_parallel_safe(load_model_path=V1_MODEL_NAME)