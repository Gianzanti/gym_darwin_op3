import math
import os
import time
import unittest

import mujoco
import mujoco.viewer
import numpy as np
from icecream import ic

model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")

def rotation_penalty(linear_acceleration, rotation_threshold=0.1):
    ic(linear_acceleration)

    # Calculate the gravity vector (assuming z-axis is upwards)
    gravity_vector = np.array([0.0, 0.0, 9.81]) 

    # Calculate the difference between measured acceleration and gravity
    acceleration_diff = linear_acceleration - gravity_vector
    # ic(acceleration_diff)

    # Calculate the magnitude of the difference
    acceleration_diff_magnitude = np.linalg.norm(acceleration_diff)
    # ic(acceleration_diff_magnitude)

    # Calculate the projected acceleration onto the gravity vector
    projected_acceleration = np.dot(acceleration_diff, gravity_vector) / np.linalg.norm(gravity_vector) * gravity_vector
    # ic(projected_acceleration)

    # Calculate the rotational component of acceleration
    rotational_acceleration = acceleration_diff - projected_acceleration
    # ic(rotational_acceleration)

    # Calculate the rotation penalty.
    # This is a simple example, and you may need to adjust the weights and thresholds based on your specific needs.
    rotation_penalty = 0.0
    if acceleration_diff_magnitude > rotation_threshold:
        rotation_penalty = np.linalg.norm(rotational_acceleration) 
    else:
        rotation_penalty = 0.0  # No penalty for significant linear motion

    ic(rotation_penalty)
    return rotation_penalty

class TestModel(unittest.TestCase):
    def test_model_properties(self):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        print ("Model properties")
        print ("nq (number of generalized coordinates = dim(qpos)): ", model.nq)
        print ("nv (number of degrees of freedom = dim(qvel)): ", model.nv)
        print ("nu (number of actuators/controls = dim(ctrl)): ", model.nu)
        print ("nbody (number of bodies): ", model.nbody)
        print ("njnt (number of joints): ", model.njnt)
        print ("ngeom (number of geoms): ", model.ngeom)
        print ("nsite (number of sites): ", model.nsite)
        print ("nsensor (number of sensors): ", model.nsensor)
        print ("nsensordata: ", model.nsensordata)
        # print ("nuserdata: ", model.nuserdata)
        # print ("nqpos: ", model.nqpos)
        # print ("nqvel: ", model.nqvel)
        # print ("nqacc: ", model.nqacc)
        # print ("nqfrc: ", model.nqfrc)

        print ("Data properties")
        print ("qpos: ", data.qpos)
        print ("qvel: ", data.qvel)
        print ("qacc: ", data.qacc)
        print ("qfrc_applied: ", data.qfrc_applied)
        

# typedef enum mjtJoint_ {          // type of degree of freedom
#   mjJNT_FREE          = 0,        // global position and orientation (quat)       (7)
#   mjJNT_BALL,                     // orientation (quat) relative to parent        (4)
#   mjJNT_SLIDE,                    // sliding distance along body-fixed axis       (1)
#   mjJNT_HINGE                     // rotation angle (rad) around body-fixed axis  (1)
# } mjtJoint;


    def test_viewer(self):
        model = mujoco.MjModel.from_xml_path(model_path)
        self.assertIsNotNone(model)

        data = mujoco.MjData(model)
        self.assertIsNotNone(data)

        # set initial state
        mujoco.mj_resetData(model, data)

        target_left_shoulder = 1.30
        target_left_elbow = -0.30
        target_right_shoulder = -1.30
        target_right_elbow = 0.30

        target_left_hip = -0.35
        target_left_knee = 0.50
        target_left_ankle = 0.25

        target_right_hip = 0.35
        target_right_knee = -0.50
        target_right_ankle = -0.25
        
        velocity = 0.05

        done = False

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            while viewer.is_running() and time.time() - start < 60:
                step_start = time.time()

                # if data.ctrl[1] < target_left_shoulder: # 1.30
                #     data.ctrl[1] += velocity

                # if data.ctrl[2] > target_left_elbow: # -0.30
                #     data.ctrl[2] -= velocity

                # if data.ctrl[4] > target_right_shoulder: # -1.30
                #     data.ctrl[4] -= velocity

                # if data.ctrl[5] < target_right_elbow: # 0.30
                #     data.ctrl[5] += velocity


                # if data.ctrl[8] > target_left_hip: # -0.25
                #     data.ctrl[8] -= velocity
                
                # if data.ctrl[9] < target_left_knee: # 0.50
                #     data.ctrl[9] += velocity

                # if data.ctrl[10] < target_left_ankle: # 0.25
                #     data.ctrl[10] += velocity

                # if data.ctrl[14] < target_right_hip: # 0.25
                #     data.ctrl[14] += velocity
                
                # if data.ctrl[15] > target_right_knee: # -0.50
                #     data.ctrl[15] -= velocity

                # if data.ctrl[16] > target_right_ankle: # -0.25
                #     data.ctrl[16] -= velocity

                # print sensor data
                # print(data.sensordata)
                # if not done:
                print (f"qpos: {data.qpos[2]}")
                # print (f"Z Axis: {data.qpos[2]}")
                # print (f"Orientation: {data.qpos[3:7]}")
                # print (f"Gyro: {data.sensordata[3:6]}")
                # print (f"Acceleration: {data.sensordata[0:3]}")
                    # print (f"qvel: {data.qvel[2:]}")
                    # print(f"Inertia: {data.cinert[1:]}")
                    # print(f"Velocity: {data.cvel[1:]}")
                    # print(f"Actuator forces: {data.qfrc_actuator[6:]}")
                    # print(f"Sensor data: {data.sensordata}")
                    # if data.sensordata[2] > 5:
                    #     done = True


                
                # flying = 
                # ic(np.sum(data.cfrc_ext))
                # if 
                # ic(data.cfrc_ext)

                # angle_rotation = 2 * math.acos(data.qpos[3])
                # print(f"Angle Rotation: {angle_rotation}")
                # penalty = 1.25 * math.pow(angle_rotation, 2)
                # print(f"Penalty Orientation: {penalty}")
                # rotation_penalty(data.sensordata[0:3])


                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


