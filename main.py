import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
data = mujoco.MjData(model)


time_step = 0.0001
model.opt.timestep = time_step  

with mujoco.viewer.launch_passive(model, data) as viewer:
    robot1_joint1_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint1")
    robot1_joint2_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint2")
    robot1_joint3_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint3")
    robot1_joint4_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint4")
    robot1_joint5_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint5")
    robot1_ghost_steer_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:ghost-steer")
    robot1_drive_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:drive")
    
    data.ctrl[robot1_drive_index] = 3
    print(data.ctrl[robot1_drive_index])

    while viewer.is_running():
        step_count = 0
        while step_count < 10000:
            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1
        data.ctrl[robot1_drive_index] = 0
        mujoco.mj_step(model, data)
        viewer.sync()