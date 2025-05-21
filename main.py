import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController, Direction, Layer
from object_remover import remove_object_on_plane
from object_placer import place_object_on_table
import threading
import time

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

def start_object_remover_thread(model, data, object_joint_ids):
    # plane parameters
    plane_positions = [[2.8, 1.0],[2.8, -1.0]]
    lower_plane_radius = 0.23
    lower_plane_z = 0.23

    threading.Thread(
        target=remove_object_on_plane,
        args=(model, data, plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
        daemon=True
    ).start()

def start_object_placer_thread(model, data, object_joint_ids):
    # object positions parameters
    left_object_position = [1, -2.5, 0.28]
    right_object_position = [-1, -2.5, 0.28]

    threading.Thread(
        target=place_object_on_table,
        args=(model, data, left_object_position, right_object_position, object_joint_ids),
        daemon=True
    ).start()

def main():
    model, data = get_data_and_model()

    object_ids = get_object_ids(model)
    object_joint_ids = []
    for i in object_ids:
        joint_name = f"object{i}:joint"
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            object_joint_ids.append((i, joint_id))
        except Exception:
            print(f"Joint {joint_name} not found in main thread")

    # Start the asynchronous thread
    start_object_remover_thread(model, data, object_joint_ids)
    start_object_placer_thread(model, data, object_joint_ids)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller = MirobotController(viewer, model, data)

        controller.origin_position_to_picking_position(direction_flag = Direction.RIGHT, render_flag = True)
        controller.execute_pick_motion(direction_flag = Direction.RIGHT, render_flag = True)
        controller.picking_position_to_pre_placing_position(render_flag = True)
        controller.rotate_joint1_to_front(render_flag = True)
        controller.pre_placing_position_to_placing_position(direction_flag = Direction.LEFT, render_flag = True)
        controller.placing_at_lower_layer(render_flag = True)
        controller.placing_position_to_pre_origin_position(render_flag = True)
        controller.placing_position_to_origin_position(render_flag = True)
        controller.reset_all_joints(render_flag = True)

        controller.origin_position_to_picking_position(direction_flag = Direction.LEFT, render_flag = True)
        controller.execute_pick_motion(direction_flag = Direction.LEFT, render_flag = True)
        controller.picking_position_to_pre_placing_position(render_flag = True)
        controller.rotate_joint1_to_front(render_flag = True)
        controller.pre_placing_position_to_placing_position(direction_flag = Direction.RIGHT, render_flag = True)
        controller.placing_at_lower_layer(render_flag = True)
        controller.placing_position_to_pre_origin_position(render_flag = True)
        controller.placing_position_to_origin_position(render_flag = True)
        controller.reset_all_joints(render_flag = True)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()