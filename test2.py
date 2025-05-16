import mujoco
import mujoco.viewer
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
data = mujoco.MjData(model)

# 设置仿真步长
time_step = 0.0001
model.opt.timestep = time_step  # 将自定义步长应用到模型

# print model.geom()
# print(model.actuator.motor('forward'))

# 启动被动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit viewer...")
    last_time = time.time()
    frame_count = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            # print(f"Simulated FPS: {frame_count}")
            frame_count = 0
            last_time = now