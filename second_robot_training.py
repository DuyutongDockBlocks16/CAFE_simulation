import gymnasium as gym
import sec_robot_env 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
import mujoco.viewer
import time
import os
from datetime import datetime

gym.register(
    id="SecondRobotMuJoCoEnv-v0",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={"xml_path": "scene_mirobot.xml"}
)

APPROACHING_MODEL_NAME = "ppo_mujoco_continued_2050K.zip"

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=10):
        super().__init__()
        self.env = env
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        return True

def approach_model_training(env):
    # Create PPO model with training parameters
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=3e-4,     # Learning rate
                n_steps=2048,           # Collect 2048 steps of experience each time
                batch_size=64,          # Process 64 samples per batch
                tensorboard_log="./ppo_logs/")  # Log save path
    

    save_interval = 50_000 
    total_steps = 1_600_000_000
    
    for i in range(total_steps // save_interval):  
        model.learn(total_timesteps=save_interval,      
                   callback=RenderCallback(env),       # Render callback
                   reset_num_timesteps=False)          # Don't reset timestep counter
        
        current_steps = (i + 1) * save_interval
        model.save(f"ppo_mujoco_car_{current_steps // 1000}K")
        print(f"Saved model at {current_steps:,} steps (ppo_mujoco_car_{current_steps // 1000}K.zip)")
    
    env.close()

def approach_model_training(env, load_model_path=None):
    
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
                    n_steps=2048,           # Collect 2048 steps of experience each time
                    batch_size=64,          # Process 64 samples per batch
                    tensorboard_log="./ppo_logs/")  # Log save path
        loaded_steps = 0

    save_interval = 50_000 
    total_additional_steps = 10_000_000  
    
    print(f"🚀 Starting training...")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Save interval: {save_interval:,} steps")
    print(f"   Model type: {'Continued' if load_model_path else 'New'}")
    
    num_iterations = total_additional_steps // save_interval
    
    for i in range(num_iterations):
        print(f"\n--- Training Progress: {i+1}/{num_iterations} ---")
        
        model.learn(total_timesteps=save_interval,      
                   callback=RenderCallback(env),       # Render callback
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

def make_env(rank, seed=0):
    """Factory function to create environment"""
    def _init():
        env = gym.make("SecondRobotMuJoCoEnv-v0")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# def approach_model_training_parallel():
#     # Create 4 parallel environments
#     num_envs = 4
#     env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

#     env = VecMonitor(env)
    
#     # Create a PPO model
#     model = PPO("MlpPolicy", env, verbose=1,
#                 learning_rate=3e-4,
#                 n_steps=512,        # 512 steps per environment
#                 batch_size=64,      # Batch size
#                 tensorboard_log="./ppo_logs/")
    
#     # Training loop
#     for i in range(20):
#         # 1M steps will be distributed to 4 environments, 250k steps each
#         model.learn(total_timesteps=1_000_000,
#                    reset_num_timesteps=False)
        
#         # Save the same model (not 4 models)
#         model.save(f"ppo_mujoco_parallel_{i+1}M")
#         print(f"Saved single model at {(i+1)}M steps")

#     env.close()

def approach_model_training_parallel():
    # Create 4 parallel environments
    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)
    
    # Create a PPO model
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=512,        # 512 steps per environment
                batch_size=64,      # Batch size
                tensorboard_log="./ppo_logs/")

    save_interval = 50_000     
    total_steps = 20_000_000
    
    # Training loop
    num_iterations = total_steps // save_interval  
    
    for i in range(num_iterations):
        print(f"\n--- Parallel Training Progress: {i+1}/{num_iterations} ---")
        
        # 50K steps will be distributed to 4 environments, 12.5K steps each
        model.learn(total_timesteps=save_interval,
                   reset_num_timesteps=False)
        
        current_steps = (i + 1) * save_interval
        model_name = f"ppo_mujoco_parallel_{current_steps // 1000}K"
        model.save(model_name)
        print(f"💾 Saved: {model_name}.zip ({current_steps:,} total steps)")
        
        if current_steps % 1_000_000 == 0:
            millions = current_steps // 1_000_000
            print(f"🎉 Milestone: Reached {millions}M steps!")

    env.close()
    print(f"\n🎊 Parallel training completed! Total steps: {total_steps:,}")

def continue_training_with_backup():
    """Continue training with backup - using 4 parallel environments"""
    # Create 4 parallel environments (consistent with approach_model_training_parallel)
    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    try:
        # Check model file
        if not os.path.exists(APPROACHING_MODEL_NAME):
            print(f"❌ Model {APPROACHING_MODEL_NAME} not found!")
            return
        
        # Backup original model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}_{APPROACHING_MODEL_NAME}"
        os.system(f"cp {APPROACHING_MODEL_NAME} {backup_name}")
        print(f"📁 Created backup: {backup_name}")
        
        # Load model (using same environment configuration)
        model = PPO.load(APPROACHING_MODEL_NAME, env=env)
        print(f"✅ Loaded {APPROACHING_MODEL_NAME} with 4 parallel environments")
        
        # Reset tensorboard log
        model.tensorboard_log = f"./ppo_logs/experiment_202506121741"
        
        # Continue training for 10M steps (10 iterations, 1M steps each)
        print("🚀 Starting continued training for 10M steps with 4 parallel environments...")
        
        for i in range(10):
            print(f"\n--- Training batch {i+1}/10 ---")
            model.learn(total_timesteps=1_000_000,      # 1M steps per iteration distributed to 4 environments, 250k steps each
                       reset_num_timesteps=False)       # Remove tensorboard_log parameter
            
            # Save intermediate model every 1M steps
            intermediate_name = f"ppo_mujoco_parallel_{10+i+1}M"
            model.save(intermediate_name)
            print(f"✅ Saved intermediate model: {intermediate_name}.zip")
        
        # Display training statistics
        print("\n📊 Training Summary:")
        print(f"Parallel environments: {num_envs}")
        print(f"Original model: {APPROACHING_MODEL_NAME}")
        print(f"Additional steps: 10,000,000 (2.5M per environment)")
        print(f"Final model: ppo_mujoco_parallel_20M.zip")
        print(f"Total training: 20M steps")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        env.close()

def continue_training_from_10000K():
    
    model_to_load = "ppo_mujoco_car_10000K.zip"
    
    env = gym.make("SecondRobotMuJoCoEnv-v0")
    
    try:
        if not os.path.exists(model_to_load):
            print(f"❌ Model {model_to_load} not found!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}_{model_to_load}"
        os.system(f"cp {model_to_load} {backup_name}")
        print(f"📁 Created backup: {backup_name}")
        
        model = PPO.load(model_to_load, env=env)
        print(f"✅ Successfully loaded {model_to_load}")
        print(f"   Starting from 10,000K (10M) steps")
        
        model.tensorboard_log = f"./ppo_logs/continued_from_10000K_{timestamp}/"
        
        save_interval = 50_000      
        total_additional_steps = 5_000_000  
        starting_step_count = 10_000_000    
        
        print(f"🚀 Continuing training from 10M steps...")
        print(f"   Additional steps: {total_additional_steps:,}")
        print(f"   Save interval: {save_interval:,} steps")
        print(f"   Final target: {starting_step_count + total_additional_steps:,} steps")
        
        num_iterations = total_additional_steps // save_interval  
        
        for i in range(num_iterations):
            print(f"\n--- Continuing Training Progress: {i+1}/{num_iterations} ---")
            
            model.learn(total_timesteps=save_interval,
                       callback=RenderCallback(env),
                       reset_num_timesteps=False) 
            
            current_total_steps = starting_step_count + (i + 1) * save_interval
            model_name = f"ppo_mujoco_car_{current_total_steps // 1000}K"
            model.save(model_name)
            
            print(f"💾 Saved: {model_name}.zip ({current_total_steps:,} total steps)")
            
            if (i + 1) * save_interval % 1_000_000 == 0:
                millions = (current_total_steps) // 1_000_000
                print(f"🎉 Milestone: Reached {millions}M total steps!")
        
        final_steps = starting_step_count + total_additional_steps
        final_model_name = f"ppo_mujoco_car_{final_steps // 1000}K_final"
        model.save(final_model_name)
        
        print(f"\n🎊 ============ CONTINUED TRAINING COMPLETED! ============")
        print(f"📊 Training Summary:")
        print(f"   Original model: {model_to_load}")
        print(f"   Starting steps: {starting_step_count:,}")
        print(f"   Additional steps: {total_additional_steps:,}")
        print(f"   Final total steps: {final_steps:,}")
        print(f"   Final model: {final_model_name}.zip")
        print(f"   Models saved: {num_iterations + 1}")
        
    except Exception as e:
        print(f"❌ Error during continued training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

def approach_model_implementation(env):
    model = PPO.load(APPROACHING_MODEL_NAME, env=env)
    obs, info = env.reset()
    for _ in range(200000000000):
        env.render()  # Render at every step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # obs, info = env.reset()
            env.unwrapped.data.ctrl[:] = 0
            mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  
            break
    
    # env.close()


if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    # approach_model_training(approach_env, load_model_path=APPROACHING_MODEL_NAME)
    approach_model_training(approach_env)
    # continue_training_from_10000K()
    # approach_model_training_parallel()
    # continue_training_with_backup()
    # approach_model_implementation(approach_env)
