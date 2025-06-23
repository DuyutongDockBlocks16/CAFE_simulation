import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
from util_threads.object_remover import remove_object_on_plane
from util_threads.object_placer import place_object_on_table
from config.env_config import Layer, FiniteState, Direction

class MirobotController:
    def __init__(self, model, data, left_object_position, right_object_position):
        self.model = model
        self.data = data
        self.robot1_joint1_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint1")
        self.robot1_joint2_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint2")
        self.robot1_joint3_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint3")
        self.robot1_joint4_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint4")
        self.robot1_joint5_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:Joint5")
        self.robot_1_adhere_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:adhere_winch")
        self.robot1_ghost_steer_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:ghost-steer")
        self.robot1_drive_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot1:drive")
        self.robot_1_rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1:rover")
        self.robot_origin_pos = data.xpos[self.robot_1_rover_id][:2].copy()  # [x, y]
        self.robot_origin_quat = data.xquat[self.robot_1_rover_id].copy()  # [w, x, y, z]
        self.robot_origin_yaw = self.quat_to_yaw(self.robot_origin_quat)  # [yaw]
        self.state = FiniteState.IDLE
        self.left_object_position = left_object_position
        self.right_object_position = right_object_position
        self.pick_position = None
        self.placing_position = None
        self.placing_layer = None
        self.waiting_timer = 0

    def get_status(self):
        return self.state

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
                                max_steps=100000, tol=1e-2,
                            ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0

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

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        if distance < tol:
            # print("break")
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0
            return True
        
        return False

    def pid_origin_position_to_picking_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=10.0, 
                            Kd_pos=1.0, 
                            Kp_yaw=3.0, 
                            Kd_yaw=0.1, 
                            max_steps=200000, tol=1e-3,
                            ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
    
        pos = self.data.xpos[self.robot_1_rover_id][:2]
        quat = self.data.xquat[self.robot_1_rover_id]
        yaw = self.quat_to_yaw(quat)

        direction = target_pos - pos
        distance = np.linalg.norm(direction)
        target_heading = np.arctan2(direction[1], direction[0])
        yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

        drive_ctrl = Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error))

        steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)
        

        self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -3, 3)
        self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.9, 0.9)

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        if distance < tol:
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0            
            return True
        
        return False
    
    # dirction_flag = -1 is right, dirction_flag = 1 is left
    def origin_position_to_picking_position(self, direction_flag: Direction):
        quat = [0.707, 0.0, 0.0, -0.707] 
        target_yaw = self.quat_to_yaw(quat)
        # print(f"target_yaw: {target_yaw}")
        if direction_flag == Direction.RIGHT:
            target_pos = np.array([-1, -2.3]) 
        elif direction_flag == Direction.LEFT:
            target_pos = np.array([1, -2.3]) 
        return self.pid_origin_position_to_picking_position(target_pos, target_yaw)

    def decreasing_joint3_and_joint5(self):
        self.data.ctrl[self.robot_1_adhere_index] = 1.0  # Set adhere to 1.0 to ensure the robot can pick the object
        target3, target5 = 0.425, 1.22
        current3 = self.data.ctrl[self.robot1_joint3_index]
        current5 = self.data.ctrl[self.robot1_joint5_index]
        self.data.ctrl[self.robot1_joint3_index] += 0.01 * (target3 - current3)
        self.data.ctrl[self.robot1_joint5_index] += 0.01 * (target5 - current5)
        if abs(current3 - target3) < 0.01 and abs(current5 - target5) < 0.01:
            return True
        return False

    def waiting_decreasing_joint3_and_joint5(self):
        self.waiting_timer += 1
        if self.waiting_timer > 300:
            self.waiting_timer = 0
            return True
        return False

    def joint1_turning(self, direction_flag: Direction):
        if direction_flag == Direction.RIGHT:
            target1 = 0.507
        else:
            target1 = -0.80
        current1 = self.data.ctrl[self.robot1_joint1_index]
        self.data.ctrl[self.robot1_joint1_index] += 0.01 * (target1 - current1)
        if abs(current1 - target1) < 0.01:
            return True
        return False

    def waiting_joint1_turning(self):
        self.waiting_timer += 1
        if self.waiting_timer > 300:
            self.waiting_timer = 0
            return True
        return False

    def lifting_joint3(self):
        target3 = 0.199
        current3 = self.data.ctrl[self.robot1_joint3_index]
        self.data.ctrl[self.robot1_joint3_index] += 0.01 * (target3 - current3)
        if abs(current3 - target3) < 0.01:
            return True
        return False

    def waiting_lifting_joint3(self):
        self.waiting_timer += 1
        if self.waiting_timer > 300:
            self.waiting_timer = 0
            return True
        return False

    def picking_position_to_pre_placing_position(self):
        quat = [0.707, 0.0, 0.0, -0.707] 
        target_yaw = self.quat_to_yaw(quat)
        return self.pid_picking_position_to_pre_placing_position(np.array([0, 2]) , target_yaw)

    def rotate_joint1_to_front(self, target_angle=0.0, Kp=0.0000001, tol=1e-3, max_steps=5000000):
        original_gear = self.model.actuator_gear[self.robot1_joint1_index].copy()
        
        self.model.actuator_gear[self.robot1_joint1_index] = 0.3  

        joint1_qpos_addr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot1:Joint1")

        current_angle = self.data.qpos[joint1_qpos_addr]
        error = target_angle - current_angle
        ctrl = Kp * error
        if abs(error) < 0.05:
            ctrl *= 0.2
        self.data.ctrl[self.robot1_joint1_index] = np.clip(ctrl, -0.001, 0.001)

        if abs(error) < tol:
            self.data.ctrl[self.robot1_joint1_index] = 0
            return True
        return False

    def waiting_joint1_to_front(self):
        self.waiting_timer += 1
        if self.waiting_timer > 2000:
            self.waiting_timer = 0
            return True
        return False
        
    def pid_pre_placing_position_to_placing_position(self, 
                            target_pos=np.zeros(2), 
                            target_yaw=0.0, 
                            Kp_pos=5.0, 
                            Kd_pos=0.2, 
                            Kp_yaw=1.5, 
                            Kd_yaw=0.05, 
                            max_steps=100000, tol=1e-2
                            ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0
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

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        # if distance < tol and abs(yaw_error) < tol:
        if distance < tol:
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0   
            return True

        return False

    # dirction_flag = -1 is right, dirction_flag = 1 is left
    def pre_placing_position_to_placing_position(self, direction_flag: Direction):
        quat = [1, 0.0, 0.0, 0.0] 
        target_yaw = self.quat_to_yaw(quat)
        # right
        if direction_flag == Direction.RIGHT:
            target_pos = np.array([2.51, -0.8])
        # left
        elif direction_flag == Direction.LEFT:
            target_pos = np.array([2.45, 1]) 

        return self.pid_pre_placing_position_to_placing_position(target_pos, target_yaw)

    def placing_at_current_layer(self):
        # current3 = self.data.ctrl[self.robot1_joint3_index]
        # current5 = self.data.ctrl[self.robot1_joint5_index]
        # self.data.ctrl[self.robot1_joint3_index] += 0.01 * (joint3_target - current3)
        # self.data.ctrl[self.robot1_joint5_index] += 0.01 * (joint5_target - current5)
        # if abs(current3 - joint3_target) < tol and abs(current5 - joint5_target) < tol:
        #     self.data.ctrl[self.robot_1_adhere_index] = 0.0
        #     return True

        # return False
        self.data.ctrl[self.robot_1_adhere_index] = 0.0
        return True

    def placing_position_to_pre_origin_position(self):
        return self.pid_placing_position_to_pre_origin_position(np.array([-2.5, 0.0]), self.robot_origin_yaw)

    def pid_placing_position_to_pre_origin_position(self, 
                                    target_pos=np.zeros(2), 
                                    target_yaw=0.0, 
                                    Kp_pos=10.0, 
                                    Kd_pos=1.0, 
                                    Kp_yaw=3.0, 
                                    Kd_yaw=0.1, 
                                    max_steps=200000, tol=1e-3
                                ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0

        pos = self.data.xpos[self.robot_1_rover_id][:2]
        quat = self.data.xquat[self.robot_1_rover_id]
        yaw = self.quat_to_yaw(quat)

        direction = target_pos - pos
        distance = np.linalg.norm(direction)
        target_heading = np.arctan2(direction[1], direction[0])
        yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

        if abs(yaw_error) > np.pi / 2:
            drive_sign = -1
            
            yaw_error = ((target_heading + np.pi) % (2 * np.pi)) - yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        else:
            drive_sign = 1

        drive_ctrl = drive_sign * (Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error)))
        steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)

        self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -1.5, 1.5)
        self.data.ctrl[self.robot1_ghost_steer_index] = -1 * np.clip(steer_ctrl, -0.5, 0.5)

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        if distance < tol:
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0
            return True
        return False

    def placing_position_to_origin_position(self):
        return self.pid_placing_position_to_origin_position(self.robot_origin_pos, self.robot_origin_yaw)
    
    def pid_placing_position_to_origin_position(self, 
                                target_pos=np.zeros(2), 
                                target_yaw=0.0, 
                                Kp_pos=10.0, 
                                Kd_pos=1.0, 
                                Kp_yaw=3.0, 
                                Kd_yaw=0.1, 
                                max_steps=200000, tol=1e-3
                                ):
        prev_pos_error = np.zeros(2)
        prev_yaw_error = 0.0

        pos = self.data.xpos[self.robot_1_rover_id][:2]
        quat = self.data.xquat[self.robot_1_rover_id]
        yaw = self.quat_to_yaw(quat)

        direction = target_pos - pos
        distance = np.linalg.norm(direction)
        target_heading = np.arctan2(direction[1], direction[0])
        yaw_error = (target_heading - yaw + np.pi) % (2 * np.pi) - np.pi

        
        if abs(yaw_error) > np.pi / 2:
            drive_sign = -1
            
            yaw_error = ((target_heading + np.pi) % (2 * np.pi)) - yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        else:
            drive_sign = 1

        drive_ctrl = drive_sign * (Kp_pos * distance + Kd_pos * (distance - np.linalg.norm(prev_pos_error)))
        steer_ctrl = Kp_yaw * yaw_error + Kd_yaw * (yaw_error - prev_yaw_error)

        self.data.ctrl[self.robot1_drive_index] = np.clip(drive_ctrl, -1.5, 1.5)
        self.data.ctrl[self.robot1_ghost_steer_index] = np.clip(steer_ctrl, -0.5, 0.5)

        prev_pos_error = direction
        prev_yaw_error = yaw_error

        if distance < tol:
            self.data.ctrl[self.robot1_drive_index] = 0
            self.data.ctrl[self.robot1_ghost_steer_index] = 0
            return True
        return False

    def reset_all_joints(self):
        self.data.ctrl[self.robot1_joint1_index] = 0
        self.data.ctrl[self.robot1_joint2_index] = 0
        self.data.ctrl[self.robot1_joint3_index] = 0
        self.data.ctrl[self.robot1_joint4_index] = 0
        self.data.ctrl[self.robot1_joint5_index] = 0
        self.data.ctrl[self.robot_1_adhere_index] = 0.0
        self.data.ctrl[self.robot1_ghost_steer_index] = 0.0
        self.data.ctrl[self.robot1_drive_index] = 0.0
        # all gears sets to 1.0
        for joint_index in [self.robot1_joint1_index, self.robot1_joint2_index, self.robot1_joint3_index,
                            self.robot1_joint4_index, self.robot1_joint5_index]:
            self.model.actuator_gear[joint_index] = 1.0
        return True

    def pid_joint2_joint3_to_upper_layer(self, joint2_target=-0.02735, joint3_target=-0.078, gear=0.05, tol=1e-3):
        idx2 = self.robot1_joint2_index
        idx3 = self.robot1_joint3_index


        self.model.actuator_gear[idx2] = gear
        self.model.actuator_gear[idx3] = gear


        current2 = self.data.ctrl[idx2]
        current3 = self.data.ctrl[idx3]


        self.data.ctrl[idx2] += 0.01 * ((joint2_target / gear) - current2)
        self.data.ctrl[idx3] += 0.01 * ((joint3_target / gear) - current3)

        if abs(current2 - (joint2_target / gear)) < tol and abs(current3 - (joint3_target / gear)) < tol:
            return True
        return False

    def waiting_pid_joint2_joint3_to_upper_layer(self):
        self.waiting_timer += 1
        if self.waiting_timer > 700:
            self.waiting_timer = 0
            return True
        return False


    def pid_joint3_to_upper_layer(self, targets={"robot1:Joint2": 0.050}, gear=0.05, tol=1e-3):
        finished = True
        for joint_name, target_pos in targets.items():
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            self.model.actuator_gear[idx] = gear
            current = self.data.ctrl[idx]
            self.data.ctrl[idx] += 0.01 * ((target_pos / gear) - current)
            if abs(current - (target_pos / gear)) >= tol:
                finished = False
        # self.data.ctrl[self.robot_1_adhere_index] = 0.0  # Ensure the robot can pick the object
        return finished

    def waiting_pid_joint3_to_upper_layer(self):
        self.waiting_timer += 1
        if self.waiting_timer > 1800:
            self.waiting_timer = 0
            return True
        return False

    def step(self, current_object_position):
        if self.state == FiniteState.IDLE:
            self.pick_position = current_object_position
            self.state = FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION
        elif self.state == FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION and np.allclose(self.pick_position, self.left_object_position):
            finished = self.origin_position_to_picking_position(direction_flag=Direction.LEFT)
            if finished:
                self.state = FiniteState.JOINT1_TURNING 
        elif self.state == FiniteState.ORIGIN_POSITION_TO_PICKING_POSITION and np.allclose(self.pick_position, self.right_object_position):
            finished = self.origin_position_to_picking_position(direction_flag=Direction.RIGHT)
            if finished:
                self.state = FiniteState.JOINT1_TURNING
        elif self.state == FiniteState.JOINT1_TURNING and np.allclose(self.pick_position, self.left_object_position):
            finished = self.joint1_turning(direction_flag=Direction.LEFT)
            if finished:
                self.state = FiniteState.WAITING_JOINT1_TURNING
        elif self.state == FiniteState.JOINT1_TURNING and np.allclose(self.pick_position, self.right_object_position):
            finished = self.joint1_turning(direction_flag=Direction.RIGHT)
            if finished:
                self.state = FiniteState.WAITING_JOINT1_TURNING
        elif self.state == FiniteState.WAITING_JOINT1_TURNING:
            finished = self.waiting_joint1_turning()
            if finished:
                self.state = FiniteState.DECREASING_JOINT3_AND_JOINT5
        elif self.state == FiniteState.DECREASING_JOINT3_AND_JOINT5:
            finished = self.decreasing_joint3_and_joint5()
            if finished:
                self.state = FiniteState.WAITING_DECREASING_JOINT3_AND_JOINT5
        elif self.state == FiniteState.WAITING_DECREASING_JOINT3_AND_JOINT5:
            finished = self.waiting_decreasing_joint3_and_joint5()
            if finished:
                self.state = FiniteState.LIFTING_JOINT3
        elif self.state == FiniteState.LIFTING_JOINT3:
            finished = self.lifting_joint3()
            if finished:
                self.state = FiniteState.WAITING_LIFTING_JOINT3
        elif self.state == FiniteState.WAITING_LIFTING_JOINT3:
            finished = self.waiting_lifting_joint3()
            if finished:
                self.state = FiniteState.PICKING_POSITION_TO_PRE_PLACING_POSITION
        elif self.state == FiniteState.PICKING_POSITION_TO_PRE_PLACING_POSITION:
            finished = self.picking_position_to_pre_placing_position()
            if finished:
                self.state = FiniteState.ROTATING_JOINT1_TO_FRONT
        elif self.state == FiniteState.ROTATING_JOINT1_TO_FRONT:
            finished = self.rotate_joint1_to_front()
            if finished:
                self.state = FiniteState.WAITING_ROTATING_JOINT1_TO_FRONT
        elif self.state == FiniteState.WAITING_ROTATING_JOINT1_TO_FRONT:
            finished = self.waiting_joint1_to_front()
            if finished:
                self.state = FiniteState.PRE_PLACING_POSITION_TO_PLACING_POSITION
                self.placing_position = random.choice([Direction.LEFT, Direction.RIGHT])
        elif self.state == FiniteState.PRE_PLACING_POSITION_TO_PLACING_POSITION and self.placing_position == Direction.LEFT:
            finished = self.pre_placing_position_to_placing_position(direction_flag=Direction.LEFT)
            if finished:
                self.state = FiniteState.PLACING_AT_LAYER
                self.placing_layer = random.choices(
                    [Layer.LOWER, Layer.UPPER],
                    weights=[0.5, 0.5] 
                )[0]
                # self.placing_layer = Layer.UPPER
        elif self.state == FiniteState.PRE_PLACING_POSITION_TO_PLACING_POSITION and self.placing_position == Direction.RIGHT: 
            finished = self.pre_placing_position_to_placing_position(direction_flag=Direction.RIGHT)
            if finished:
                self.state = FiniteState.PLACING_AT_LAYER
                self.placing_layer = random.choices(
                    [Layer.LOWER, Layer.UPPER],
                    weights=[0.5, 0.5] 
                )[0]  
                # self.placing_layer = Layer.UPPER
        elif self.state == FiniteState.PLACING_AT_LAYER and self.placing_layer == Layer.LOWER:
            finished = self.placing_at_current_layer()
            if finished:
                self.state = FiniteState.PLACING_POSITION_TO_PRE_ORIGIN_POSITION
        elif self.state == FiniteState.PLACING_AT_LAYER and self.placing_layer == Layer.UPPER:
            finished = self.pid_joint2_joint3_to_upper_layer()
            if finished:
                self.state = FiniteState.WAITING_JOINT2_JOINT3_TO_UPPER_LAYER
        elif self.state == FiniteState.WAITING_JOINT2_JOINT3_TO_UPPER_LAYER:
            finished = self.waiting_pid_joint2_joint3_to_upper_layer()
            if finished:
                self.state = FiniteState.JOINT3_TO_UPPER_LAYER
        elif self.state == FiniteState.JOINT3_TO_UPPER_LAYER:
            finished = self.pid_joint3_to_upper_layer()
            if finished:
                self.state = FiniteState.WAITING_JOINT3_TO_UPPER_LAYER
        elif self.state == FiniteState.WAITING_JOINT3_TO_UPPER_LAYER:
            finished = self.waiting_pid_joint3_to_upper_layer()
            if finished:
                self.state = FiniteState.PLACING_AT_UPPER_LAYER
        elif self.state == FiniteState.PLACING_AT_UPPER_LAYER:
            finished = self.placing_at_current_layer()
            if finished:
                self.state = FiniteState.PLACING_POSITION_TO_PRE_ORIGIN_POSITION
        elif self.state == FiniteState.PLACING_POSITION_TO_PRE_ORIGIN_POSITION:
            finished = self.placing_position_to_pre_origin_position()
            if finished:
                self.state = FiniteState.PLACING_POSITION_TO_ORIGIN_POSITION
        elif self.state == FiniteState.PLACING_POSITION_TO_ORIGIN_POSITION:
            finished = self.placing_position_to_origin_position()
            if finished:
                self.state = FiniteState.RESETTING_ALL_JOINTS
        elif self.state == FiniteState.RESETTING_ALL_JOINTS:
            finished = self.reset_all_joints()
            if finished:
                self.state = FiniteState.IDLE
                self.pick_position = None
                self.placing_position = None
                self.placing_layer = None
                self.waiting_timer = 0
    
    def set_state(self, state: FiniteState):
        self.state = state