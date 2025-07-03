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
import numpy as np

gym.register(
    id="SecondRobotMuJoCoEnv-v0",
    entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
    kwargs={
        "xml_path": "xml/scene_mirobot.xml",
    }
)

# def approach_model_training(env):
#     # Create PPO model with training parameters
#     model = PPO("MlpPolicy", env, verbose=1, 
#                 learning_rate=3e-4,     # Learning rate
#                 n_steps=2048,           # Collect 2048 steps of experience each time
#                 batch_size=64,          # Process 64 samples per batch
#                 tensorboard_log="./ppo_logs/")  # Log save path
    

#     save_interval = 50_000 
#     total_steps = 1_600_000_000
    
#     for i in range(total_steps // save_interval):  
#         model.learn(total_timesteps=save_interval,      
#                    callback=RenderCallback(env),       # Render callback
#                    reset_num_timesteps=False)          # Don't reset timestep counter
        
#         current_steps = (i + 1) * save_interval
#         model.save(f"ppo_mujoco_car_{current_steps // 1000}K")
#         print(f"Saved model at {current_steps:,} steps (ppo_mujoco_car_{current_steps // 1000}K.zip)")
    
#     env.close()

def approach_model_training(env, load_model_path=None):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    combined_callback = CallbackList([
        # RenderCallback(env),
        SuccessCheckpointCallback("./checkpoints"),
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

def make_env(rank, seed=0):
    """Factory function to create environment"""
    def _init():
        env = gym.make(
            "SecondRobotMuJoCoEnv-v0",
            action_repeat=4
            )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

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

    env.render()
    sleep(15)

    for _ in range(200000000000):
        env.render()  # Render at every step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # obs, info = env.reset()
            env.unwrapped.data.ctrl[:] = 0
            mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)  
            break

    model = env.unwrapped.model
    data = env.unwrapped.data

    env.close()

    # sleep 1s 
    # sleep(10)

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

def model_fine_tune(env, load_model_path=None):
    model = PPO.load(
        load_model_path,
        env = env,
        ent_coef = 0.0005,
        clip_range = 0.05
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )

    combined_callback = CallbackList([
        # RenderCallback(env),
        SuccessCheckpointCallback("./fine_tune_checkpoints"),
        episode_collector
    ]) 

    model.learn(
        total_timesteps=20_000_000,
        # total_timesteps=200_000,
        reset_num_timesteps=False,  
        tb_log_name="fine_tuning", 
        callback=combined_callback
    )

    model.save("best_model.zip")

def approach_model_training_generalization(env, load_model_path):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/episode_data_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )

    combined_callback = CallbackList([
        RenderCallback(env),
        SuccessCheckpointCallback("./fine_tune_checkpoints"),
        episode_collector
    ]) 
    
    # Load existing model
    model = PPO.load(load_model_path, env=env)
    model.learning_rate = 2e-4  # 🎯 Lower learning rate to protect learned knowledge
    model.ent_coef = 0.02       # 🎯 Increase exploration to adapt to new targets
    
    # Extract trained steps
    import re
    match = re.search(r'(\d+)K', load_model_path)
    loaded_steps = int(match.group(1)) * 1000 if match else 0

    # Target position list
    target_positions = [
        [2, -2],        # Original target
        [2.2, -1.8],    # Slight offset
        [1.8, -2.2],    # Slight offset
        [2.5, -1.5],    # Medium offset
        [1.5, -2.5],    # Medium offset
        [2, -1.5],      # Y-axis offset
        [1.5, -2],      # X-axis offset
        [1.5, 1.5],     # Different quadrant
        [2, 1],         # Upper target
        [0, 0],         # Center target
    ]
    
    save_interval = 50_000
    total_additional_steps = 100_000_000  # 100M steps for generalization training
    num_iterations = total_additional_steps // save_interval
    
    target_change_interval = 10  # Change target every 10 iterations
    
    print(f"  Starting generalization training...")
    print(f"   Base model: {load_model_path}")
    print(f"   Number of target positions: {len(target_positions)}")
    print(f"   Training steps: {total_additional_steps:,}")
    
    for i in range(num_iterations):
        #  Periodically change target position
        if i % target_change_interval == 0:
            current_target = target_positions[i // target_change_interval % len(target_positions)]
            env.unwrapped.target_position_x_y = current_target
            print(f"\n Switching target to: {current_target}")
        
        print(f"\n--- Generalization training progress: {i+1}/{num_iterations} ---")
        print(f"   Current target: {env.unwrapped.target_position_x_y}")
        
        model.learn(total_timesteps=save_interval, 
                   callback=combined_callback, 
                   reset_num_timesteps=False)
        
        current_total_steps = loaded_steps + (i + 1) * save_interval
        
        #  Test generalization ability every 20 iterations
        if i % 20 == 0 and i > 0:
            print(f"\n🔍 Generalization ability test:")
            generalization_results = test_multiple_targets(model, env, target_positions[:5])
            avg_success = np.mean(list(generalization_results.values()))
            print(f"   Average success rate: {avg_success:.1%}")
            
            if avg_success < 0.4:
                print("⚠️ Generalization ability declining, adjusting training parameters...")
                model.learning_rate *= 0.8
                model.ent_coef = min(0.1, model.ent_coef * 1.5)
        
        # Save model
        model_name = f"generalized_model_{current_total_steps // 1000}K"
        model.save(model_name)
        print(f"💾 Saved: {model_name}.zip")
    
    # Restore original target
    env.unwrapped.target_position_x_y = [2, -2]
    
    return model

def test_multiple_targets(model, env, test_targets, tests_per_target=3):
    """Test performance on multiple target positions"""

    results = {}
    original_target = env.unwrapped.target_position_x_y.copy()
    
    for target in test_targets:
        env.unwrapped.target_position_x_y = target
        successes = 0
        
        for _ in range(tests_per_target):
            obs, _ = env.reset()
            
            for step in range(5000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                
                if reward > 1000:
                    successes += 1
                    break
                    
                if done or truncated:
                    break
        
        success_rate = successes / tests_per_target
        results[str(target)] = success_rate
        print(f"   Target {target}: {success_rate:.1%}")

    # Restore original target
    env.unwrapped.target_position_x_y = original_target
    
    return results


if __name__ == "__main__":
    approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    approach_model_training(approach_env, load_model_path=APPROACHING_MODEL_NAME)
    # approach_model_training(approach_env)
    # continue_training_from_10000K()
    # approach_model_training_parallel()
    # continue_training_with_backup()
    # approach_model_implementation(approach_env)
    # fine_tuned_model = model_fine_tune(approach_env, APPROACHING_MODEL_NAME)

# if __name__ == "__main__":
#     approach_env = gym.make("SecondRobotMuJoCoEnv-v0")
    
#     #  Using trained model for generalization training
#     existing_model_path = "original_place_to_another_corner.zip"  # Your trained model
    
#     if os.path.exists(existing_model_path):
#         print(" Starting generalization training...")
#         generalized_model = approach_model_training_generalization(
#             approach_env, existing_model_path
#         )
        
#         # Final test
#         print("\n Final generalization ability test:")
#         final_targets = [[2, -2], [2.2, -1.8], [1.5, 1.5], [0, 0], [2.5, -1.5]]
#         final_results = test_multiple_targets(generalized_model, approach_env, final_targets, tests_per_target=10)
        
#         avg_final_success = np.mean(list(final_results.values()))
#         print(f"🎊 Final average generalization success rate: {avg_final_success:.1%}")
        
#     else:
#         print(f"❌ Model file not found: {existing_model_path}")