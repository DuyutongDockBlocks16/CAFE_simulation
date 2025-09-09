import mujoco
import random

def randomize_materials_at_runtime(model):
    colors = {
        'yellow': [1.0, 1.0, 0.2, 1.0],  # yellow
        'red': [1.0, 0.2, 0.2, 1.0],     # red
    }
    
    color_assignment = ['red'] * 10
    
    yellow_indices = random.sample(range(10), 2) 
    
    for idx in yellow_indices:
        color_assignment[idx] = 'yellow'
    
    
    for i in range(10):
        try:
            geom_name = f"object{i}_geom"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            
            if geom_id >= 0:
                color_name = color_assignment[i]
                color_rgba = colors[color_name]
                model.geom_rgba[geom_id] = color_rgba
                # print(f"object {geom_name}'s color set to: {color_rgba}")
        except:
            print(f"object {geom_name} not found")