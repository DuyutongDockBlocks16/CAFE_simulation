import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController, Direction, Layer, FiniteState
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
import threading
import time
import numpy as np
import random

def get_data_and_model():
    model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
    data = mujoco.MjData(model)
    time_step = 0.005
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

def start_object_remover_threads(model, data, object_joint_ids):
    # lower plane parameters
    lower_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
    lower_plane_radius = 0.23
    lower_plane_z = 0.23

    threading.Thread(
        target=remove_object_on_plane,
        args=(model, data, lower_plane_positions, lower_plane_radius, lower_plane_z, object_joint_ids),
        daemon=True
    ).start()

    # upper plane parameters
    upper_plane_positions = [[2.8, 1.0],[2.8, -1.0]]
    upper_plane_radius = 0.15
    upper_plane_z = 0.43

    threading.Thread(
        target=remove_object_on_plane,
        args=(model, data, upper_plane_positions, upper_plane_radius, upper_plane_z, object_joint_ids),
        daemon=True
    ).start()


def start_object_placer_thread(model, data, object_joint_ids, left_object_position, right_object_position, shared_state):
    # object positions parameters
    threading.Thread(
        target=place_object_on_table,
        args=(model, data, left_object_position, right_object_position, object_joint_ids),
        kwargs={"shared_state": shared_state},
        daemon=True
    ).start()

def main():
    left_object_position = [1, -2.5, 0.28]
    right_object_position = [-1, -2.5, 0.28]

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
    start_object_remover_threads(model, data, object_joint_ids)

    shared_state = {"current_object_index": None, "current_object_position": None, "stop": False, "stopped": True}
    start_object_placer_thread(model, data, object_joint_ids, left_object_position, right_object_position, shared_state)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller = MirobotController(model, data, left_object_position, right_object_position)
        

        while True:
            status = controller.get_status()
            # print(f"Current object index: {shared_state['current_object_index']}")
            if shared_state["current_object_index"] >= len(object_joint_ids) and status == FiniteState.IDLE:
                print("All objects have been placed. Exit")
                break

            controller.step(shared_state["current_object_position"])

            mujoco.mj_step(model, data)

            if not np.all(np.isfinite(data.qacc)) or np.any(np.abs(data.qacc) > 1e7):
                print("QACC error detected! Simulation unstable, exiting loop.")
                break

            viewer.sync()

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()