import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController

model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
data = mujoco.MjData(model)

time_step = 0.001
model.opt.timestep = time_step  

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller = MirobotController(viewer, model, data)
    controller.origin_position_to_picking_position(-1)
    controller.execute_pick_motion(-1)
    controller.picking_position_to_pre_placing_position()
    controller.rotate_joint1_to_front()
    controller.pre_placing_position_to_placing_position(1,1)
    controller.placing_at_lower_layer()
    controller.placing_position_to_origin_position()

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
