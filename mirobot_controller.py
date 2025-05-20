import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

class MirobotController:
    def __init__(self, viewer, model, data):
        self.viewer = viewer
        self.model = model
        self.data = data
        self.robot1_joint1_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint1")
        self.robot1_joint2_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint2")
        self.robot1_joint3_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint3")
        self.robot1_joint4_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint4")
        self.robot1_joint5_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint5")
        self.robot1_ghost_steer_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:ghost-steer")
        self.robot1_drive_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:drive")
        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_origin_pos = data.xpos[self.robot_1_rover_id][:2].copy()  # [x, y]
        self.robot_origin_quat = data.xquat[self.robot_1_rover_id].copy()  # [w, x, y, z]
        self.robot_origin_yaw = self.quat_to_yaw(self.robot_origin_quat)  # [yaw]

    def quat_to_yaw(self, quat):
        # MuJoCo: [w, x, y, z] -> scipy: [x, y, z, w]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        _, _, yaw = r.as_euler('xyz', degrees=False)
        return yaw

    def pid_picking_position_to_pre_placing_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=5.0, 
                            Kd_pos=0.2, 
                            Kp_yaw=1.5, 
                            Kd_yaw=0.05, 
                            max_steps=100000, tol=1e-2):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
        for _ in range(max_steps):
            pos = self.data.xpos[self.robot_1_rover_id][:2]
            quat = self.data.xquat[self.robot_1_rover_id]
            yaw = self.quat_to_yaw(quat)

            direction = target_pos - pos
            # print(f"pos: {pos}, target_pos: {target_pos}, direction: {direction}")
            distance = np.linalg.norm(direction)
            target_heading = np.arctan2(direction[1], direction[0])
            yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

            drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))

            steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)
            

            self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, 1.5, -1.5)
            self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.2, 0.2)

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            prev_pos_error = direction
            prev_yaw_error = yaw_error

            # print(f"drive_ctrl: {drive_ctrl}, steer_ctrl: {steer_ctrl}, distance: {distance}, yaw_error: {yaw_error}")

            # if distance < tol and abs(yaw_error) < tol:
            if distance < tol:
                # print("break")
                break

        self.data.ctrl[self.robot1_drive_index] = 0
        self.data.ctrl[self.robot1_ghost_steer_index] = 0

    def pid_origin_position_to_picking_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=10.0, 
                            Kd_pos=1.0, 
                            Kp_yaw=3.0, 
                            Kd_yaw=0.1, 
                            max_steps=200000, tol=1e-3):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
        for _ in range(max_steps):
            pos = self.data.xpos[self.robot_1_rover_id][:2]
            quat = self.data.xquat[self.robot_1_rover_id]
            yaw = self.quat_to_yaw(quat)

            direction = target_pos - pos
            # print(f"pos: {pos}, target_pos: {target_pos}, direction: {direction}")
            distance = np.linalg.norm(direction)
            target_heading = np.arctan2(direction[1], direction[0])
            yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

            drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))
            # if abs(yaw_error) < 0.2:
            #     drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))
                
            # else:
            #     drive_ctrl = 0

            steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)
            

            self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -3, 3)
            self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.9, 0.9)

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            prev_pos_error = direction
            prev_yaw_error = yaw_error

            # print(f"drive_ctrl: {drive_ctrl}, steer_ctrl: {steer_ctrl}, distance: {distance}, yaw_error: {yaw_error}")

            # if distance < tol and abs(yaw_error) < tol:
            if distance < tol:
                break

        self.data.ctrl[self.robot1_drive_index] = 0
        self.data.ctrl[self.robot1_ghost_steer_index] = 0
    
    # dirction_flag = -1 is right, dirction_flag = 1 is left
    def origin_position_to_picking_position(self, direction_flag):
        quat = [0.707, 0.0, 0.0, -0.707] 
        target_yaw = self.quat_to_yaw(quat)
        # print(f"target_yaw: {target_yaw}")
        if direction_flag == -1:
            target_pos = np.array([-1, -2.3]) 
        else:
            target_pos = np.array([1, -2.3]) 
        self.pid_origin_position_to_picking_position(target_pos, target_yaw)

    # dirction_flag = -1 is right, dirction_flag = 1 is left
    def execute_pick_motion(self, direction_flag):
        self.data.ctrl[self.robot1_joint3_index] = 1.1
        self.data.ctrl[self.robot1_joint5_index] = -0.64

        step_count = 0
        while step_count < 1000:
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            step_count += 1
        
        if direction_flag == -1:
            self.data.ctrl[self.robot1_joint1_index] = 0.507
        elif direction_flag == 1:
            self.data.ctrl[self.robot1_joint1_index] = -0.80
            
        step_count = 0
        while step_count < 1000:
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            step_count += 1

        self.data.ctrl[self.robot1_joint3_index] = 0.674
        step_count = 0
        while step_count < 3000:
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            step_count += 1

    def picking_position_to_pre_placing_position(self):
        quat = [0.707, 0.0, 0.0, -0.707] 
        target_yaw = self.quat_to_yaw(quat)
        self.pid_picking_position_to_pre_placing_position(np.array([0, 2]) , target_yaw)

    def rotate_joint1_to_front(self, target_angle=0.0, Kp=0.5, tol=1e-3, max_steps=5000):
        joint1_qpos_addr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot1:Joint1")
        step_count = 0
        while step_count < max_steps:
            current_angle = self.data.qpos[joint1_qpos_addr]
            error = target_angle - current_angle
            ctrl = Kp * error
            if abs(error) < 0.05:
                ctrl *= 0.2
            self.data.ctrl[self.robot1_joint1_index] = np.clip(ctrl, -0.05, 0.05)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            if abs(error) < tol:
                break
            step_count += 1
        self.data.ctrl[self.robot1_joint1_index] = 0

    def pid_pre_placing_position_to_placing_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=5.0, 
                            Kd_pos=0.2, 
                            Kp_yaw=1.5, 
                            Kd_yaw=0.05, 
                            max_steps=100000, tol=1e-2):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
        for _ in range(max_steps):
            pos = self.data.xpos[self.robot_1_rover_id][:2]
            quat = self.data.xquat[self.robot_1_rover_id]
            yaw = self.quat_to_yaw(quat)

            direction = target_pos - pos
            # print(f"pos: {pos}, target_pos: {target_pos}, direction: {direction}")
            distance = np.linalg.norm(direction)
            target_heading = np.arctan2(direction[1], direction[0])
            yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

            drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))

            steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)
            

            self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -3, 3)
            self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.9, 0.9)

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            prev_pos_error = direction
            prev_yaw_error = yaw_error

            # print(f"drive_ctrl: {drive_ctrl}, steer_ctrl: {steer_ctrl}, distance: {distance}, yaw_error: {yaw_error}")

            # if distance < tol and abs(yaw_error) < tol:
            if distance < tol:
                # print("break")
                break

        self.data.ctrl[self.robot1_drive_index] = 0
        self.data.ctrl[self.robot1_ghost_steer_index] = 0

    # dirction_flag = -1 is right, dirction_flag = 1 is left
    def pre_placing_position_to_placing_position(self, direction_flag, layer_flag):
        quat = [1, 0.0, 0.0, 0.0] 
        target_yaw = self.quat_to_yaw(quat)
        # right
        if direction_flag == -1:
            target_pos = np.array([2.45, -1]) 
        # left
        else:
            target_pos = np.array([2.45, 1]) 
            
        self.pid_pre_placing_position_to_placing_position(target_pos, target_yaw)

    def placing_at_lower_layer(self, joint3_target=0.674, joint5_target=0.4, hold_steps=3000):
        self.data.ctrl[self.robot1_joint3_index] = joint3_target
        self.data.ctrl[self.robot1_joint5_index] = joint5_target
        for _ in range(hold_steps):
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def placing_position_to_origin_position(self):
        self.pid_placing_position_to_origin_position(self.robot_origin_pos, self.robot_origin_yaw)

    def pid_placing_position_to_origin_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=10.0, 
                            Kd_pos=1.0, 
                            Kp_yaw=3.0, 
                            Kd_yaw=0.1, 
                            max_steps=200000, tol=1e-3):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
        for _ in range(max_steps):
            pos = self.data.xpos[self.robot_1_rover_id][:2]
            quat = self.data.xquat[self.robot_1_rover_id]
            yaw = self.quat_to_yaw(quat)

            direction = target_pos - pos
            # print(f"pos: {pos}, target_pos: {target_pos}, direction: {direction}")
            distance = np.linalg.norm(direction)
            target_heading = np.arctan2(direction[1], direction[0])
            yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

            drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))
            # if abs(yaw_error) < 0.2:
            #     drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))
                
            # else:
            #     drive_ctrl = 0

            steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)
            

            self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, 3, -3)
            self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.4, 0.4)

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            prev_pos_error = direction
            prev_yaw_error = yaw_error

            # print(f"drive_ctrl: {drive_ctrl}, steer_ctrl: {steer_ctrl}, distance: {distance}, yaw_error: {yaw_error}")

            # if distance < tol and abs(yaw_error) < tol:
            if distance < tol:
                break

        self.data.ctrl[self.robot1_drive_index] = 0
        self.data.ctrl[self.robot1_ghost_steer_index] = 0