import gymnasium as gym
import sac_second_robot_env 
from stable_baselines3 import HerReplayBuffer, SAC
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
    id="SacSecondRobotMuJoCoEnv-v0",
    entry_point="sac_second_robot_env:SacSecondRobotMuJoCoEnv",
    kwargs={"xml_path": "xml/scene_mirobot.xml"}
)

n_sampled_goal = 4

def approach_model_training(env, load_model_path=None):
    """
    HER-SAC训练函数，支持从检查点继续训练
    
    Args:
        env: 训练环境
        load_model_path: 可选，加载已有模型的路径
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./logs/episode_data_her_sac_{timestamp}.jsonl"
    
    episode_collector = EpisodeBatchCollector(
        output_file=output_file,
        batch_size=5,
        verbose=1
    )
    
    combined_callback = CallbackList([
        RenderCallback(env),
        SuccessCheckpointCallback("./checkpoints"),
        episode_collector
    ])

    # 🎯 检查是否从已有模型加载
    if load_model_path and os.path.exists(f"{load_model_path}.zip"):
        print(f"🔄 Loading existing HER-SAC model from: {load_model_path}")
        model = SAC.load(load_model_path, env=env)
        
        # 获取已训练步数
        loaded_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
        print(f"   ✅ Model loaded successfully!")
        print(f"   📊 Previous training steps: {loaded_steps:,}")
        
    else:
        print("🆕 Creating new HER-SAC model...")
        model = SAC(
            "MultiInputPolicy", 
            env, 
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            learning_starts=6000,  # 🔄 确保大于max_episode_length
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            tensorboard_log="./her_sac_logs/"  # 🔄 添加tensorboard日志
        )
        loaded_steps = 0

    # 🎯 训练配置
    save_interval = 50_000  # 每50K步保存一次
    total_additional_steps = 100_000_000  # 总额外训练步数（100M步）

    print(f"🚀 Starting HER-SAC training...")
    print(f"   Algorithm: HER + SAC")
    print(f"   Policy: MultiInputPolicy (Dict observation space)")
    print(f"   n_sampled_goal: {n_sampled_goal}")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Save interval: {save_interval:,} steps")
    print(f"   Model type: {'Continued' if load_model_path else 'New'}")
    print(f"   Learning starts: {model.learning_starts:,} steps")
    
    num_iterations = total_additional_steps // save_interval
    
    # 🎯 分批训练循环
    for i in range(num_iterations):
        print(f"\n--- HER-SAC Training Progress: {i+1}/{num_iterations} ---")
        current_iteration_start = time.time()
        
        # 🔄 学习一个批次
        model.learn(
            total_timesteps=save_interval,      
            callback=combined_callback, 
            reset_num_timesteps=False,  # 不重置时间步计数器
            # progress_bar=True  # 显示进度条
        )
        
        current_total_steps = loaded_steps + (i + 1) * save_interval
        iteration_time = time.time() - current_iteration_start
        
        # 🎯 保存模型检查点
        if load_model_path:
            model_name = f"her_sac_continued_{current_total_steps // 1000}K"
        else:
            model_name = f"her_sac_highway_{current_total_steps // 1000}K"
            
        model.save(model_name)
        
        # 🎯 计算统计信息
        steps_per_second = save_interval / iteration_time
        estimated_remaining_time = (num_iterations - i - 1) * iteration_time / 3600  # 小时
        
        print(f"💾 Saved checkpoint: {model_name}.zip")
        print(f"📊 Progress: {current_total_steps:,} total steps")
        print(f"⏱️  Iteration time: {iteration_time:.1f}s ({steps_per_second:.1f} steps/s)")
        print(f"🕐 Estimated remaining: {estimated_remaining_time:.1f} hours")
        
        # 🎯 里程碑提示
        if current_total_steps % 500_000 == 0:
            millions = current_total_steps / 1_000_000
            print(f"🎉 Milestone: Reached {millions:.1f}M total steps!")
            
            # 🔍 可选：在里程碑处评估模型性能
            if hasattr(episode_collector, 'episode_buffer') and len(episode_collector.episode_buffer) > 0:
                recent_episodes = episode_collector.episode_buffer[-20:]  # 最近20个episode
                if recent_episodes:
                    avg_reward = np.mean([ep.get('total_reward', -5000) for ep in recent_episodes])
                    avg_steps = np.mean([ep.get('total_steps', 5000) for ep in recent_episodes])
                    success_count = sum(1 for ep in recent_episodes if ep.get('is_success', False))
                    success_rate = success_count / len(recent_episodes)
                    
                    print(f"📈 Recent Performance (last {len(recent_episodes)} episodes):")
                    print(f"   Average Reward: {avg_reward:.1f}")
                    print(f"   Average Steps: {avg_steps:.1f}")
                    print(f"   Success Rate: {success_rate:.1%}")
    
    # 🎯 训练完成，保存最终模型
    final_total_steps = loaded_steps + total_additional_steps
    if load_model_path:
        final_model_name = f"her_sac_continued_{final_total_steps // 1000}K_final"
    else:
        final_model_name = f"her_sac_highway_{final_total_steps // 1000}K_final"
    
    model.save(final_model_name)
    
    # 🎯 训练总结
    print(f"\n🎊 ============ HER-SAC TRAINING COMPLETED! ============")
    print(f"📊 Training Summary:")
    print(f"   Algorithm: HER + SAC")
    if load_model_path:
        print(f"   Original model: {load_model_path}")
        print(f"   Starting steps: {loaded_steps:,}")
    else:
        print(f"   Training type: New model from scratch")
        print(f"   Starting steps: 0")
    print(f"   Additional steps: {total_additional_steps:,}")
    print(f"   Final total steps: {final_total_steps:,}")
    print(f"   Final model: {final_model_name}.zip")
    print(f"   Log file: {output_file}")
    print(f"   Tensorboard logs: ./her_sac_logs/")
    
    return model

# 🎯 添加模型测试函数
def test_trained_model(model_path, env, num_episodes=10):
    """测试已训练的模型"""
    
    print(f"🧪 Testing model: {model_path}")
    model = SAC.load(model_path)
    
    success_count = 0
    total_steps = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
        if info.get('is_success', False):
            success_count += 1
            
        total_steps += steps
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}: {'✅' if info.get('is_success', False) else '❌'} "
              f"Steps: {steps}, Reward: {episode_reward:.1f}")
    
    avg_steps = total_steps / num_episodes
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\n📊 Test Results:")
    print(f"   Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Reward: {avg_reward:.1f}")
    
    return success_rate, avg_steps, avg_reward

if __name__ == "__main__":
    approach_env = gym.make("SacSecondRobotMuJoCoEnv-v0")
    
    # 🎯 选择训练模式
    load_model_path = None  # 设置为已有模型路径以继续训练，None表示从头开始
    # load_model_path = "her_sac_highway_50K"  # 例如继续训练
    
    # 🎯 开始训练
    trained_model = approach_model_training(approach_env, load_model_path)
    
    # 🎯 可选：训练完成后测试模型
    print("\n🧪 Testing final model...")
    final_model_name = f"her_sac_highway_{2000 // 1000}M_final"  # 根据实际训练步数调整
    
    try:
        test_results = test_trained_model(final_model_name, approach_env, num_episodes=5)
        print(f"Final model performance: {test_results}")
    except Exception as e:
        print(f"Could not test final model: {e}")
    
    approach_env.close()