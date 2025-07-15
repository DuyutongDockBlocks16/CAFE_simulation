import os
import pickle
import mujoco
import gymnasium as gym

def load_mujoco_state_from_file(filepath):
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

    return data

def view_saved_state(state_filepath):
    state_data = load_mujoco_state_from_file(state_filepath)
    
    gym.register(
        id="SecondRobotMuJoCoEnv-v0",
        entry_point="sec_robot_env:SecondRobotMuJoCoEnv",
        kwargs={
            "xml_path": "xml/scene_mirobot.xml",
        }
    )
    print(f"🔍 Loading and viewing state from: {state_filepath}")
    
    env = gym.make("SecondRobotMuJoCoEnv-v0")
    env.reset()
    
    mujoco_model = env.unwrapped.model
    mujoco_data = env.unwrapped.data
    
    restore_mujoco_state(mujoco_model, mujoco_data, state_data)

    # 🎯 打印所有object joints的位置
    print("\n📦 Object joints positions:")
    print("=" * 50)
    
    # 🎯 查找所有object joints
    object_joints = []
    for i in range(mujoco_model.njnt):
        joint_name = mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name.startswith("object") and joint_name.endswith(":joint"):
            try:
                object_id = int(joint_name.split("object")[1].split(":")[0])
                object_joints.append((object_id, i, joint_name))
            except (ValueError, IndexError):
                continue
    
    # 🎯 按object_id排序
    object_joints.sort(key=lambda x: x[0])
    
    # 🎯 打印每个object joint的位置
    for object_id, joint_id, joint_name in object_joints:
        # 获取joint对应的body位置
        body_id = mujoco_model.jnt_bodyid[joint_id]
        body_name = mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        # 获取位置信息
        position = mujoco_data.xpos[body_id]
        quaternion = mujoco_data.xquat[body_id]
        
        print(f"Object {object_id:2d} ({joint_name}):")
        print(f"   Body: {body_name}")
        print(f"   Position: [{position[0]:8.5f}, {position[1]:8.5f}, {position[2]:8.5f}]")
        print(f"   Quaternion: [{quaternion[0]:6.3f}, {quaternion[1]:6.3f}, {quaternion[2]:6.3f}, {quaternion[3]:6.3f}]")
        print()
    
    # 🎯 打印统计信息
    print(f"📊 Total objects found: {len(object_joints)}")
    
    print("=" * 50)

    env.close()
    
    print("Starting viewer...")
    
    try:
        with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
            
            step_count = 0
            while viewer.is_running():
                # 可选：运行少量物理步骤保持场景活跃
                # if step_count % 100 == 0:  # 每100帧运行一次物理
                #     mujoco.mj_step(mujoco_model, mujoco_data)
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