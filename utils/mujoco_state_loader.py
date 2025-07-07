import os
import pickle

def load_mujoco_state_from_file(filepath):
    """
    从文件加载MuJoCo状态
    
    Args:
        filepath: 状态文件路径
    
    Returns:
        dict: 状态数据字典，如果失败返回None
    """
    try:
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        print(f"📁 State loaded from: {filepath}")
        print(f"   Simulation time: {state_data['time']:.3f}")
        print(f"   Robot position: {state_data['qpos'][:3]}")
        
        return state_data
        
    except FileNotFoundError:
        print(f"❌ State file not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        return None