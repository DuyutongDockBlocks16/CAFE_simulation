import mujoco
import mujoco.viewer
from mirobot_controller import MirobotController, Direction, Layer

model = mujoco.MjModel.from_xml_path("scene_mirobot.xml")
data = mujoco.MjData(model)

time_step = 0.001
model.opt.timestep = time_step  

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller = MirobotController(viewer, model, data)

    controller.origin_position_to_picking_position(direction_flag = Direction.RIGHT, render_flag = False)
    controller.execute_pick_motion(direction_flag = Direction.RIGHT, render_flag = False)
    controller.picking_position_to_pre_placing_position(render_flag = False)
    controller.rotate_joint1_to_front(render_flag = False)
    controller.pre_placing_position_to_placing_position(direction_flag = Direction.LEFT, render_flag = False)
    controller.placing_at_lower_layer(render_flag = False)
    controller.placing_position_to_pre_origin_position(render_flag = False)
    controller.placing_position_to_origin_position(render_flag = False)
    controller.reset_all_joints(render_flag = False)

    controller.origin_position_to_picking_position(direction_flag = Direction.LEFT, render_flag = False)
    controller.execute_pick_motion(direction_flag = Direction.LEFT, render_flag = False)
    controller.picking_position_to_pre_placing_position(render_flag = False)
    controller.rotate_joint1_to_front(render_flag = False)
    controller.pre_placing_position_to_placing_position(direction_flag = Direction.RIGHT, render_flag = False)
    controller.placing_at_lower_layer(render_flag = False)
    controller.placing_position_to_pre_origin_position(render_flag = False)
    controller.placing_position_to_origin_position(render_flag = False)
    controller.reset_all_joints(render_flag = False)

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
