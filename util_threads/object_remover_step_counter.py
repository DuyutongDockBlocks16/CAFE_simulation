import time
import random
import threading

def remove_object_on_plane_with_step_counter(model, data, plane_positions, plane_radius, plane_z, 
                                           object_joint_ids, check_interval=0.5):
    print("remover thread running (step-based)")
    removed_ids = set()
    
    pending_removals = {}  # {object_id: {'target_step': int, 'qpos_adr': int, 'joint_name': str}}
    step_counter = 0

    def is_on_plane(obj_pos, plane_pos, plane_radius, plane_z, z_tol=0.05):
        dx = obj_pos[0] - plane_pos[0]
        dy = obj_pos[1] - plane_pos[1]
        dz = abs(obj_pos[2] - plane_z)
        return (dx**2 + dy**2) <= plane_radius**2 and dz < z_tol

    def schedule_removal_by_steps(object_id, qpos_adr, joint_name, min_delay_steps=7, max_delay_steps=15):
        delay_steps = random.randint(min_delay_steps, max_delay_steps)
        target_step = step_counter + delay_steps
        
        pending_removals[object_id] = {
            'target_step': target_step,
            'qpos_adr': qpos_adr,
            'joint_name': joint_name,
            'delay_steps': delay_steps
        }
        
        delay_seconds = delay_steps * model.opt.timestep
        print(f"{joint_name} scheduled for removal at step {target_step} (in {delay_steps} steps, ~{delay_seconds:.2f}s)")

    def process_removals_by_steps():
        to_remove = []

        # print(f"Processing removals at step {step_counter}, pending: {len(pending_removals)}")
        
        for object_id, removal_info in pending_removals.items():
            if step_counter >= removal_info['target_step']:
                qpos_adr = removal_info['qpos_adr']
                joint_name = removal_info['joint_name']
                
                data.qpos[qpos_adr] = 5
                data.qpos[qpos_adr+1] = 0
                data.qpos[qpos_adr+2] = 1.1
                data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                
                print(f"{joint_name} removed at step {step_counter}")
                to_remove.append(object_id)
        

        for object_id in to_remove:
            del pending_removals[object_id]

    while True:

        step_counter += 1
        

        process_removals_by_steps()
        

        for i, joint_id in object_joint_ids:
            if i in removed_ids or i in pending_removals:
                continue
                
            joint_name = f"object{i}:joint"
            qpos_adr = model.jnt_qposadr[joint_id]
            obj_pos = data.qpos[qpos_adr : qpos_adr+3]
            
            if is_on_plane(obj_pos, plane_positions[0], plane_radius, plane_z) or \
               is_on_plane(obj_pos, plane_positions[1], plane_radius, plane_z):
                print(f"{joint_name} is on plane, scheduling removal...")
                removed_ids.add(i)
                schedule_removal_by_steps(i, qpos_adr, joint_name)

        time.sleep(check_interval)
        
def remove_object_on_plane_with_step_counter_with_flag(model, data, plane_positions, plane_radius, plane_z, 
                                           object_joint_ids, shared_state, check_interval=0.5, min_delay_steps=7, max_delay_steps=15):
    print("remover thread running (step-based)")
    removed_ids = set()
    
    pending_removals = {}  # {object_id: {'target_step': int, 'qpos_adr': int, 'joint_name': str}}
    step_counter = 0

    def is_on_plane(obj_pos, plane_pos, plane_radius, plane_z, z_tol=0.05):
        dx = obj_pos[0] - plane_pos[0]
        dy = obj_pos[1] - plane_pos[1]
        dz = abs(obj_pos[2] - plane_z)
        return (dx**2 + dy**2) <= plane_radius**2 and dz < z_tol

    def schedule_removal_by_steps(object_id, qpos_adr, joint_name, min_delay_steps=7, max_delay_steps=15):
        delay_steps = random.randint(min_delay_steps, max_delay_steps)
        target_step = step_counter + delay_steps
        
        pending_removals[object_id] = {
            'target_step': target_step,
            'qpos_adr': qpos_adr,
            'joint_name': joint_name,
            'delay_steps': delay_steps
        }
        
        delay_seconds = delay_steps * model.opt.timestep
        print(f"{joint_name} scheduled for removal at step {target_step} (in {delay_steps} steps, ~{delay_seconds:.2f}s)")

    def process_removals_by_steps():
        to_remove = []

        # print(f"Processing removals at step {step_counter}, pending: {len(pending_removals)}")
        
        for object_id, removal_info in pending_removals.items():
            if step_counter >= removal_info['target_step']:
                qpos_adr = removal_info['qpos_adr']
                joint_name = removal_info['joint_name']
                
                data.qpos[qpos_adr] = 5
                data.qpos[qpos_adr+1] = 0
                data.qpos[qpos_adr+2] = 1.1
                data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                
                print(f"{joint_name} removed at step {step_counter}")
                to_remove.append(object_id)
        

        for object_id in to_remove:
            del pending_removals[object_id]

    while True:
        if shared_state.get("should_stop", False):
            print("🛑 remover thread stopped by shared_state")
            break

        step_counter += 1
        
        process_removals_by_steps()

        for i, joint_id in object_joint_ids:
            if i in removed_ids or i in pending_removals:
                continue
                
            joint_name = f"object{i}:joint"
            qpos_adr = model.jnt_qposadr[joint_id]
            obj_pos = data.qpos[qpos_adr : qpos_adr+3]
            
            if is_on_plane(obj_pos, plane_positions[0], plane_radius, plane_z) or \
               is_on_plane(obj_pos, plane_positions[1], plane_radius, plane_z):
                print(f"{joint_name} is on plane, scheduling removal...")
                removed_ids.add(i)
                schedule_removal_by_steps(i, qpos_adr, joint_name, min_delay_steps, max_delay_steps)

        time.sleep(check_interval)