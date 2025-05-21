import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController, Direction, Layer
from object_remover import remove_object_on_plane
import threading
import time
import random

def get_data_and_model():
    model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
    data = mujoco.MjData(model)
    time_step = 0.001
    model.opt.timestep = time_step  
    return model, data

def get_object_ids(model):
    object_ids = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.startswith("object") and name.endswith(":joint"):
            # extract the N from the name
            try:
                num = int(name.split(":")[0][6:])  # "objectN:joint" -> N
                object_ids.append(num)
            except Exception:
                continue
    return sorted(object_ids)

def start_remover_thread(model, data, plane_positions, lower_plane_radius, lower_plane_z, joint_ids):
    threading.Thread(
        target=remove_object_on_plane,
        args=(model, data, plane_positions, lower_plane_radius, lower_plane_z, joint_ids),
        daemon=True
    ).start()

def main():
    model, data = get_data_and_model()

    object_ids = get_object_ids(model)

    joint_ids = []
    for i in object_ids:
        joint_name = f"object{i}:joint"
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_ids.append((i, joint_id))
        except Exception:
            print(f"Joint {joint_name} not found in main thread")

    # plane parameters
    plane_positions = [[2.8, 1.0],[2.8, -1.0]]
    lower_plane_radius = 0.23
    lower_plane_z = 0.23

    # Start the asynchronous thread
    start_remover_thread(model, data, plane_positions, lower_plane_radius, lower_plane_z, joint_ids)

    

    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller = MirobotController(viewer, model, data)

        object_id = 0
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"object{object_id}:joint")
        qpos_adr = model.jnt_qposadr[joint_id]
        # Left
        data.qpos[qpos_adr : qpos_adr+3] = [1, -2.5, 0.28]

        object_id = 1
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"object{object_id}:joint")
        qpos_adr = model.jnt_qposadr[joint_id]
        # Right
        data.qpos[qpos_adr : qpos_adr+3] = [-1, -2.5, 0.28]

        # controller.origin_position_to_picking_position(direction_flag = Direction.RIGHT, render_flag = False)
        # controller.execute_pick_motion(direction_flag = Direction.RIGHT, render_flag = False)
        # controller.picking_position_to_pre_placing_position(render_flag = False)
        # controller.rotate_joint1_to_front(render_flag = False)
        # controller.pre_placing_position_to_placing_position(direction_flag = Direction.LEFT, render_flag = False)
        # controller.placing_at_lower_layer(render_flag = True)
        # controller.placing_position_to_pre_origin_position(render_flag = True)
        # controller.placing_position_to_origin_position(render_flag = True)
        # controller.reset_all_joints(render_flag = True)

        controller.origin_position_to_picking_position(direction_flag = Direction.LEFT, render_flag = False)
        controller.execute_pick_motion(direction_flag = Direction.LEFT, render_flag = False)
        controller.picking_position_to_pre_placing_position(render_flag = False)
        controller.rotate_joint1_to_front(render_flag = False)
        controller.pre_placing_position_to_placing_position(direction_flag = Direction.RIGHT, render_flag = False)
        controller.placing_at_lower_layer(render_flag = True)
        controller.placing_position_to_pre_origin_position(render_flag = True)
        controller.placing_position_to_origin_position(render_flag = True)
        controller.reset_all_joints(render_flag = True)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()