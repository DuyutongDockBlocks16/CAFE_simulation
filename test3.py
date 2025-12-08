
import mujoco
import mujoco.viewer
from utils.mujoco_state_loader import load_mujoco_state_from_file

def load_and_view(xml_path, state_filepath):
    """加载状态并启动MuJoCo查看器"""
    print(f"📁 加载模型: {xml_path}")
    print(f"📁 加载状态: {state_filepath}")
    
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 加载状态
    state_data = load_mujoco_state_from_file(state_filepath)
    data.qpos[:] = state_data['qpos']
    data.qvel[:] = state_data['qvel'] 
    data.ctrl[:] = state_data['ctrl']
    
    # 前向运动学计算
    mujoco.mj_forward(model, data)
    
    print("✅ 状态加载完成")
    print("🎮 启动MuJoCo查看器 - 您可以手动控制关节")
    print("💡 按ESC退出查看器")
    
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            # 不进行物理步进，保持静态状态
            viewer.sync()

if __name__ == "__main__":
    # 🔧 修改这里的路径
    xml_path = "xml/scene_mirobot.xml"  # 您的XML文件路径
    state_filepath = "./saved_states/robot_state_20250825_153503.pkl"  # 您的状态文件路径
    
    load_and_view(xml_path, state_filepath)