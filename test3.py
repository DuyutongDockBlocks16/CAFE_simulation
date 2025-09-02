import numpy as np

def analyze_alignment_reward_performance():
    """分析你的对齐奖励函数在不同情况下的表现"""
    
    standard_quat = np.array([-0.707, 0.0, 0.707, 0.0])
    reward_weight = 1.0  # 你的 self.reward_weights["alignment"]
    
    # 测试不同的真空吸嘴朝向
    test_cases = {
        "完美对齐": np.array([-0.707, 0.0, 0.707, 0.0]),      # 与标准完全相同
        "四元数取负": np.array([0.707, 0.0, -0.707, 0.0]),     # -q (相同旋转)
        "轻微偏差": np.array([-0.70, 0.05, 0.71, 0.02]),      # 小幅偏差
        "45度偏差": np.array([-0.50, 0.0, 0.866, 0.0]),       # 45度差异
        "90度偏差": np.array([0.0, 0.0, 1.0, 0.0]),           # 90度差异
        "135度偏差": np.array([0.383, 0.0, 0.924, 0.0]),      # 135度差异
        "完全相反": np.array([0.707, 0.0, 0.707, 0.0]),       # 180度差异
        "随机朝向1": np.array([0.5, 0.5, 0.5, 0.5]),          # 随机方向
        "随机朝向2": np.array([0.866, 0.5, 0.0, 0.0]),        # 另一随机方向
        "垂直向下": np.array([0.0, 0.0, 0.0, 1.0]),           # Z轴向下
    }
    
    print("🎯 对齐奖励函数表现分析")
    print("=" * 60)
    print(f"{'情况':12s} {'点积':>8s} {'绝对值':>8s} {'奖励':>8s} {'角度差':>8s}")
    print("-" * 60)
    
    for name, test_quat in test_cases.items():
        # 你的函数计算
        dot_product = np.dot(test_quat, standard_quat)
        abs_dot = np.abs(dot_product)
        precision_reward = abs_dot * reward_weight
        
        # 计算实际角度差
        angle_rad = 2 * np.arccos(np.clip(abs_dot, 0, 1))
        angle_deg = np.degrees(angle_rad)
        
        print(f"{name:12s} {dot_product:8.3f} {abs_dot:8.3f} {precision_reward:8.3f} {angle_deg:7.1f}°")

analyze_alignment_reward_performance()