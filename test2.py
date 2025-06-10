import mujoco
import mujoco.viewer
import time


model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
# model = mujoco.MjModel.from_xml_path("active_adhesion.xml")
data = mujoco.MjData(model)


time_step = 0.001
model.opt.timestep = time_step  



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
            
            frame_count = 0
            last_time = now