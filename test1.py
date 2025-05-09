import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("./venv/lib/python3.11/site-packages/gymnasium_robotics/envs/assets/fetch/pick_and_place.xml")
data = mujoco.MjData(model)

# 创建 viewer 窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Running simulation...")

    t0 = time.time()
    while viewer.is_running():
        step_start = time.time()

        # 示例：控制 gripper 开闭（注意 joint 名需匹配 XML）
        # 举例：夹紧 gripper
        data.ctrl[model.actuator(name="robot0:l_gripper_finger_joint").id] = 1.0
        data.ctrl[model.actuator(name="robot0:r_gripper_finger_joint").id] = 1.0

        # 模拟步进
        mujoco.mj_step(model, data)

        # 控制帧率（可选）
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


