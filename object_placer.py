import threading
import random
import numpy as np
import time
from mirobot_controller import Direction

def place_object_on_table(model, data, left_object_position, right_object_position, object_joint_ids, shared_state, check_interval=0.5):

    placed_flag = False
    next_object_position = random.choice([left_object_position, right_object_position])
    # next_object_position = right_object_position
    current_object_position = [0, 0, 0]
    i = 0
    _, object_joint_id = object_joint_ids[i]
    qpos_adr = model.jnt_qposadr[object_joint_id]
    obj_pos = data.qpos[qpos_adr : qpos_adr+3]
    shared_state["current_object_index"] = i
    shared_state["current_object_position"] = obj_pos

    def is_another_object_near(model, data, next_object_position):
        for j in range(len(object_joint_ids)):
            if j == i:
                continue
            _, other_joint_id = object_joint_ids[j]
            other_qpos_adr = model.jnt_qposadr[other_joint_id]
            other_obj_pos = data.qpos[other_qpos_adr : other_qpos_adr+3][:2]
            if np.linalg.norm(np.array(other_obj_pos) - np.array(next_object_position)) < 0.1:
                return True
        return False

    while i < len(object_joint_ids):
        # The current_object is not in the correct position
        # print(np.allclose(obj_pos, next_object_position))
        if not np.allclose(obj_pos[:2], next_object_position[:2]) and not placed_flag:
            print(f"Placing object {i} at {next_object_position}")
            # print(f"Current object position: {obj_pos}")
            # print(f"Next object position: {next_object_position}")
            data.qpos[qpos_adr : qpos_adr+3] = next_object_position
            obj_pos = data.qpos[qpos_adr : qpos_adr+3]
            current_object_position = next_object_position
            placed_flag = True
            shared_state["current_object_position"] = current_object_position
            if np.allclose(next_object_position, left_object_position):
                next_object_position = right_object_position
                
            else:
                next_object_position = left_object_position

            
        # The current_object is moved by the robot
        if not np.allclose(obj_pos[:2], current_object_position[:2]):
            # check if there is any another object at the near of next_object_position
            # if there is, then use continue to skip current loop
            if is_another_object_near(model, data, next_object_position[:2]):
                print(f"Another object is near {next_object_position}, skipping placement")
                continue
            # if there is not, then update the next_object_position
            else:
                i += 1
                shared_state["current_object_index"] = i
                if i >= len(object_joint_ids):
                    print("All objects placed, THREAD EXIT")
                    return  
                _, object_joint_id = object_joint_ids[i]
                qpos_adr = model.jnt_qposadr[object_joint_id]
                obj_pos = data.qpos[qpos_adr : qpos_adr+3]
                placed_flag = False

        time.sleep(check_interval)