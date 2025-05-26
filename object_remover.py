import time
import random
import threading

def remove_object_on_plane(model, data, plane_positions, plane_radius, plane_z, object_joint_ids, check_interval=0.05):
    print("remover thread running")
    removed_ids = set()

    def is_on_plane(obj_pos, plane_pos, plane_radius, plane_z, z_tol=0.05):
        # print(f"Checking if object at {obj_pos} is on plane at {plane_pos}")
        dx = obj_pos[0] - plane_pos[0]
        dy = obj_pos[1] - plane_pos[1]
        dz = abs(obj_pos[2] - plane_z)
        return (dx**2 + dy**2) <= plane_radius**2 and dz < z_tol

    def delayed_remove(data, qpos_adr, joint_name, min_delay=5, max_delay=10):
        delay = random.uniform(min_delay, max_delay)
        def remove():
            data.qpos[qpos_adr+2] = -100
            print(f"{joint_name} removed from plane after {delay:.2f}s")
        threading.Timer(delay, remove).start()

    while True:
        for i, joint_id in object_joint_ids:
            # or joint_id != 0
            if i in removed_ids :
                continue
            joint_name = f"object{i}:joint"
            # print(f"Checking {joint_name}...")
            qpos_adr = model.jnt_qposadr[joint_id]
            obj_pos = data.qpos[qpos_adr : qpos_adr+3]
            # print(f"{joint_name} pos: {obj_pos}")
            if is_on_plane(obj_pos, plane_positions[0], plane_radius, plane_z) or \
                is_on_plane(obj_pos, plane_positions[1], plane_radius, plane_z):
            # if is_on_plane(obj_pos, plane_positions[1], plane_radius, plane_z):
                print(f"{joint_name} is on plane, removing...")
                removed_ids.add(i)
                delayed_remove(data, qpos_adr, joint_name)

        time.sleep(check_interval)