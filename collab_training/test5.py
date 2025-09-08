import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.insert(0, parent_dir)

import mujoco
from utils.mujoco_object_color_randomiser import randomize_materials_at_runtime
import numpy as np

from enum import Enum
import numpy as np

class ObjectColor:
    RED = np.array([1.0, 0.2, 0.2, 1.0])
    YELLOW = np.array([1.0, 1.0, 0.2, 1.0])
    
    @classmethod
    def identify(cls, color_array):
        if np.allclose(color_array, cls.RED):
            return "RED"
        elif np.allclose(color_array, cls.YELLOW):
            return "YELLOW"
        # else:
        #     return "UNKNOWN"

def _get_data_and_model():
    model = mujoco.MjModel.from_xml_path("../xml/collab_mirobot.xml")
    data = mujoco.MjData(model)
    time_step = 0.005
    model.opt.timestep = time_step  
    return model, data

# def _get_object_color(joint_id):
#     body_id = model.jnt_bodyid[joint_id]
#     geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
#     if geom_ids:
#         return model.geom_rgba[geom_ids[0]]
#     return np.zeros(4)

def _get_object_color(joint_id):
    body_id = model.jnt_bodyid[joint_id]
    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
    if geom_ids:
        color_array = model.geom_rgba[geom_ids[0]]
        color_name = ObjectColor.identify(color_array)
        return color_name
    # return "NONE"

if __name__ == "__main__":
    model, data = _get_data_and_model()
    object_joints = []
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name.startswith("object") and joint_name.endswith(":joint"):
            try:
                object_id = int(joint_name.split("object")[1].split(":")[0])
                object_joints.append((object_id, i, joint_name))
            except (ValueError, IndexError):
                continue
    randomize_materials_at_runtime(model)
    
    joint_id = object_joints[0][1]
    
    color = _get_object_color(joint_id)
    print(f"Object color for joint {joint_id}: {color}")