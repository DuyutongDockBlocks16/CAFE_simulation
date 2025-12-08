import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 20

# Define the 5 training log files
# LOG_FILES = [
#     "../logs/driver_episode_data_20250917_120511.jsonl",
#     "../logs/driver_episode_data_20250919_134832.jsonl",
#     "../logs/driver_episode_data_20250920_155212.jsonl",
#     "../logs/driver_episode_data_20250920_190030.jsonl",
#     "../logs/driver_episode_data_20250920_195253.jsonl",
#     "../logs/driver_episode_data_20250920_212641.jsonl",
#     "../logs/driver_episode_data_20250922_100256.jsonl",
#     "../logs/driver_episode_data_20250922_141502.jsonl",
#     "../logs/driver_episode_data_20250922_184926.jsonl",
# ]

LOG_FILES = [
    "../logs/driver_episode_data_20251008_192822.jsonl",
    "../logs/driver_episode_data_20251010_233052.jsonl",
    "../logs/driver_episode_data_20251012_222756.jsonl",
    "../logs/driver_episode_data_20251013_090713.jsonl",
    "../logs/driver_episode_data_20251013_134906.jsonl",
]

def read_episode_data(filename):
    episodes = []
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return episodes
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'episode_number' in data:
                    episodes.append(data)
            except json.JSONDecodeError:
                continue
    return episodes

def calculate_moving_average(data, window_size=200):
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(data[start_idx:i+1]))
    return moving_avg

def plot_reward_curves(log_files, max_episodes=None, save_path=None, reward_threshold=1000):
    plt.figure(figsize=(16, 12))
    colors = plt.cm.tab10.colors
    
    for idx, file in enumerate(log_files):
        episodes = read_episode_data(file)
        if not episodes:
            continue
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        
        episode_numbers = list(range(1, len(episodes) + 1))
        total_rewards = [ep['total_reward'] for ep in episodes]
        
        # 🔧 过滤outliers - 移除大于阈值的值
        original_count = len(total_rewards)
        filtered_rewards = []
        filtered_episodes = []
        
        for i, reward in enumerate(total_rewards):
            if reward <= reward_threshold:  # 只保留小于等于阈值的值
                filtered_rewards.append(reward)
                filtered_episodes.append(i + 1)
        
        print(f"📊 File {idx+1}: 原始 {original_count} episodes, 过滤后 {len(filtered_rewards)} episodes (移除 {original_count - len(filtered_rewards)} outliers)")
        
        # 使用过滤后的数据计算移动平均
        moving_avg_rewards = calculate_moving_average(filtered_rewards, window_size=200)
        
        plt.plot(filtered_episodes, moving_avg_rewards, color=colors[idx % 10], linewidth=2.5,
                 label=f'Training Round {idx+1}')
    
    plt.xlabel('Episode Index')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title(f'Reward Curves')
    
    plt.yscale('symlog', linthresh=1)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_reward_curves(LOG_FILES, max_episodes=12000, save_path="reward_curves_9rounds.png")
