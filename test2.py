import mujoco
import mujoco.viewer

# 加载模型
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 设置仿真步长
time_step = 0.00001
model.opt.timestep = time_step  # 将自定义步长应用到模型

# 启动被动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit viewer...")
    while viewer.is_running():
        # 手动推进仿真
        mujoco.mj_step(model, data)
        # 同步渲染
        viewer.sync()
