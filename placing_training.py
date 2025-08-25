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
from config.env_config import Direction, Layer, FiniteState
from config.training_config import PICKING_MODEL_NAME, SUCCESS_THRESHOLD
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
    id="SecondRobotPlacingMuJoCoEnv-v0",
    entry_point="placing_env:SecondRobotPlacingMuJoCoEnv",
    kwargs={
        "xml_path": "xml/scene_mirobot.xml",
        # "state_filepaths": [
        #     "saved_states/robot_state_20250726_154225.pkl", 
        #     # "saved_states/robot_state_20250721_151909.pkl"
        # ]
        # "state_filepath": "saved_states/robot_state_20250721_151909.pkl"
        # "state_filepath": "saved_states/robot_state_20250726_154225.pkl"
        # "state_filepath": "saved_states/robot_state_20250728_173657.pkl"
        # "state_filepath": "saved_states/robot_state_20250728_191655.pkl"
        "state_filepath": "saved_states/robot_state_20250825_153503.pkl"
    }
)

def make_env(rank, seed=0):
    """Factory function to create environment"""
    def _init():
        env = gym.make(
                "SecondRobotPlacingMuJoCoEnv-v0",
                action_repeat=4
            )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def placing_model_training(env, load_model_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/placing_episode_data_{timestamp}.jsonl"

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
        model = PPO("MlpPolicy", env, verbose=1, 
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
            model_name = f"placing_model_continued_{current_total_steps // 1000}K"
        else:
            model_name = f"placing_model_{current_total_steps // 1000}K"

        model.save(model_name)
        print(f"💾 Saved: {model_name}.zip ({current_total_steps:,} total steps)")
        
        if (i + 1) * save_interval % 1_000_000 == 0:
            millions = current_total_steps // 1_000_000
            print(f"🎉 Milestone: Reached {millions}M total steps!")
    
    final_total_steps = loaded_steps + total_additional_steps
    if load_model_path:
        final_model_name = f"placing_model_continued_{final_total_steps // 1000}K_final"
    else:
        final_model_name = f"placing_model_{final_total_steps // 1000}K_final"

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


def placing_model_training_parallel(load_model_path=None, num_envs=8):
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/placing_episode_data_{timestamp}.jsonl"

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

        model = PPO.load(load_model_path, env=env)

        model.ent_coef = 0.1
        model.learning_rate = 1e-4

        print(f"✅ Successfully loaded model from: {load_model_path}")
        print(f"🔄 Using {num_envs} parallel environments")

        model.tensorboard_log = f"./ppo_logs/placing_continued_parallel_{timestamp}/"

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
        
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=1e-4,     
                    n_steps=2048,           # 调整为并行环境合适的值
                    batch_size=256,          
                    n_epochs=8,            
                    ent_coef=0.02,          
                    clip_range=0.15,          
                    gae_lambda=0.95,         
                    vf_coef=1.0,            
                    # max_grad_norm=0.3,
                    tensorboard_log="./ppo_logs/")
        loaded_steps = 0

    # total_additional_steps = 16_000_000
    # total_additional_steps = 500_000
    total_additional_steps = 10_000_000

    ent_scheduler = EntCoefficientScheduler(
        initial_ent_coef=0.02,  
        # final_ent_coef=0.02,  
        # initial_ent_coef=0.1,  
        # final_ent_coef=0.1,           
        final_ent_coef=0.0005,          
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
        final_model_name = f"final_placing_model_continued_{final_total_steps // 1000}K_{timestamp}"
    else:
        final_model_name = f"final_placing_model_{final_total_steps // 1000}K_{timestamp}"

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

def placing_model_implementation(env):
    model = PPO.load(PLACING_MODEL_NAME, env=env)
    obs, info = env.reset()

    env.render()
    sleep(5)

    for _ in range(200000000000):
        env.render()  # Render at every step
        # sleep(0.1)
        action, _ = model.predict(obs, deterministic=True)
        # save all action to file
        # with open("picking_obs_log.txt", "a") as f:
        #     f.write(f"{obs.tolist()}\n")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # obs, info = env.reset()
            # env.unwrapped.data.ctrl[:] = 0
            mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  
            break

    model = env.unwrapped.model
    data = env.unwrapped.data

    env.close()

    sleep(20)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC to exit viewer...")
        last_time = time.time()
        frame_count = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                # print(f"Simulated FPS: {frame_count}")
                frame_count = 0
                last_time = now

if __name__ == "__main__":
    placing_env = gym.make("SecondRobotPlacingMuJoCoEnv-v0")
    # placing_model_training(placing_env, load_model_path=PLACING_MODEL_NAME)
    # placing_model_training(placing_env)
    placing_model_training_parallel()
    # placing_model_training_parallel(load_model_path=PLACING_MODEL_NAME)
    # placing_model_implementation(placing_env)