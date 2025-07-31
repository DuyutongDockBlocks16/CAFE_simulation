import os
import pickle
from datetime import datetime
import numpy as np

def save_mujoco_state_to_file(model, data, filename=None):
    """
    保存MuJoCo状态到文件
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        filename: 保存文件名（可选）
    
    Returns:
        str: 保存的文件路径
    """
    # 获取状态数据
    state_data = save_mujoco_state(model, data)
    
    # 生成文件名（如果未指定）
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robot_state_{timestamp}.pkl"
    
    # 确保保存目录存在
    save_dir = "saved_states"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    
    # 🎯 保存为pickle文件（保持数据精度）
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        print(f"💾 State saved to: {filepath}")
        print(f"   File size: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        return None

def save_mujoco_state(model, data):
    """
    保存 MuJoCo 状态数据
    
    Returns:
        dict: 包含所有重要状态信息的字典
    """
    state_data = {
        'qpos': data.qpos.copy(),           # 关节位置
        'qvel': data.qvel.copy(),           # 关节速度
        'ctrl': data.ctrl.copy(),           # 控制输入
        'qfrc_applied': data.qfrc_applied.copy(),  # 应用力
        'time': data.time,                  # 仿真时间
        # 可选：保存更多状态
        'qacc': data.qacc.copy(),           # 关节加速度
        'qacc_warmstart': data.qacc_warmstart.copy(),
        'qfrc_constraint': data.qfrc_constraint.copy(),
        'contact': {
            'ncon': data.ncon,
            'geom1': data.contact.geom1[:data.ncon].copy() if data.ncon > 0 else np.array([]),
            'geom2': data.contact.geom2[:data.ncon].copy() if data.ncon > 0 else np.array([]),
        }
    }
    
    print(f"💾 Saved MuJoCo state at time {data.time:.3f}")
    
    return state_data