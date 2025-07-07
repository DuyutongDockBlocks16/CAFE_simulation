import os
import pickle
import mujoco
import gymnasium as gym

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

def restore_mujoco_state(model, data, state_data):

    # 恢复主要状态
    data.qpos[:] = state_data['qpos']
    data.qvel[:] = state_data['qvel'] 
    data.ctrl[:] = state_data['ctrl']
    data.qfrc_applied[:] = state_data['qfrc_applied']
    data.time = state_data['time']
    
    mujoco.mj_forward(model, data)
    
    print(f"🔄 Restored state at time {data.time:.3f}")
    print(f"   Robot position: {data.qpos[:3]}")


def view_saved_state(state_filepath):
    print(f"🔍 Loading and viewing state from: {state_filepath}")
    
    # 🎯 步骤1: 加载状态数据
    state_data = load_mujoco_state_from_file(state_filepath)
    
    if not state_data:
        print("❌ 无法加载状态文件")
        return
    
    env = gym.make("SecondRobotMuJoCoEnv-v0")
    env.reset()
    
    mujoco_model = env.unwrapped.model
    mujoco_data = env.unwrapped.data
    
    restore_mujoco_state(mujoco_model, mujoco_data, state_data)
    
    env.close()
    

    print("Starting viewer...")
    
    try:
        with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
            
            step_count = 0
            while viewer.is_running():
                # 可选：运行少量物理步骤保持场景活跃
                if step_count % 100 == 0:  # 每100帧运行一次物理
                    mujoco.mj_step(mujoco_model, mujoco_data)
                
                viewer.sync()
                step_count += 1
                
                # 防止无限循环
                if step_count > 100000:
                    print("⏱️ 达到最大步数，退出查看器")
                    break
                    
    except Exception as e:
        print(f"❌ 查看器错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 查看器已关闭")