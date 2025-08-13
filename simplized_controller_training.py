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
import torch

gym.register(
    id="HybridMuJoCoEnv-v0",
    entry_point="hybrid_robot_env:HybridMuJoCoEnv",
    # kwargs={
    #     "xml_path": "xml/scene_mirobot.xml",
    #     "state_filepath": "saved_states/robot_state_20250723_093442.pkl"
    # }
)

def make_env(rank, seed=0):
    """Factory function to create environment"""
    def _init():
        env = gym.make(
            "HybridMuJoCoEnv-v0"
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
            model_name = f"ppo_mujoco_continued_{current_total_steps // 1000}K"
        else:
            model_name = f"ppo_mujoco_car_{current_total_steps // 1000}K"
            
        model.save(model_name)
        print(f"💾 Saved: {model_name}.zip ({current_total_steps:,} total steps)")
        
        if (i + 1) * save_interval % 1_000_000 == 0:
            millions = current_total_steps // 1_000_000
            print(f"🎉 Milestone: Reached {millions}M total steps!")
    
    final_total_steps = loaded_steps + total_additional_steps
    if load_model_path:
        final_model_name = f"ppo_mujoco_continued_{final_total_steps // 1000}K_final"
    else:
        final_model_name = f"ppo_mujoco_car_{final_total_steps // 1000}K_final"
    
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
    
if __name__ == "__main__":
    driver_env = gym.make("HybridMuJoCoEnv-v0")
    driver_model_training(driver_env)