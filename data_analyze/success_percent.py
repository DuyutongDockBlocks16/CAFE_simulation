import json
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """读取 json 文件，每行一个 episode 记录"""
    records = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_success_rate(records, window_size=10):
    """计算滑动窗口成功率"""
    successes = []
    for rec in records:
        try:
            last_reward = rec["last_5_rewards"][-1]
            success = 1 if last_reward > 59 else 0
            successes.append(success)
        except (KeyError, IndexError):
            continue

    success_rate = []
    for i in range(len(successes)):
        window = successes[max(0, i - window_size + 1):i + 1]
        rate = sum(window) / len(window)
        success_rate.append(rate)
    return success_rate

def plot_success_rate(success_rate, file_path):
    """绘制成功率曲线并保存到与 JSON 文件同一目录"""
    plt.figure(figsize=(8, 5))
    plt.plot(success_rate)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    # plt.title("Success Rate Curve (Sliding Window)")
    plt.grid(True)

    # 构造保存路径
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    save_name = base_name.replace('.jsonl', 'success_analysis_thesis.png')
    save_path = os.path.join(dir_name, save_name)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Plot saved to: {save_path}")


if __name__ == "__main__":
    file_path = "../logs/driver_episode_data_20250820_201438.jsonl"  # 你的 JSON 文件路径
    window_size = 200       # 滑动窗口大小，可以改

    records = load_data(file_path)
    success_rate = compute_success_rate(records, window_size)
    plot_success_rate(success_rate, file_path)
