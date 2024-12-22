import os
import time
import unittest

import mujoco
import mujoco.viewer

model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")

class TestModel(unittest.TestCase):
    # def test_model_properties(self):
    #     model = mujoco.MjModel.from_xml_path(model_path)
    #     data = mujoco.MjData(model)
    #     mujoco.mj_resetData(model, data)

    #     print ("Model properties")
    #     print ("nq: ", model.nq)
    #     print ("nv: ", model.nv)
    #     print ("na: ", model.na)
    #     print ("nbody: ", model.nbody)
    #     print ("njnt: ", model.njnt)
    #     print ("ngeom: ", model.ngeom)
    #     print ("nsensordata: ", model.nsensordata)
    #     print ("nuserdata: ", model.nuserdata)
    #     print ("nqpos: ", model.nqpos)
    #     print ("nqvel: ", model.nqvel)
    #     print ("nqacc: ", model.nqacc)
    #     print ("nqfrc: ", model.nqfrc)
        


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

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            while viewer.is_running() and time.time() - start < 60:
                step_start = time.time()

                if data.ctrl[1] < target_left_shoulder: # 1.30
                    data.ctrl[1] += velocity

                if data.ctrl[2] > target_left_elbow: # -0.30
                    data.ctrl[2] -= velocity

                if data.ctrl[4] > target_right_shoulder: # -1.30
                    data.ctrl[4] -= velocity

                if data.ctrl[5] < target_right_elbow: # 0.30
                    data.ctrl[5] += velocity


                if data.ctrl[8] > target_left_hip: # -0.25
                    data.ctrl[8] -= velocity
                
                if data.ctrl[9] < target_left_knee: # 0.50
                    data.ctrl[9] += velocity

                if data.ctrl[10] < target_left_ankle: # 0.25
                    data.ctrl[10] += velocity

                if data.ctrl[14] < target_right_hip: # 0.25
                    data.ctrl[14] += velocity
                
                if data.ctrl[15] > target_right_knee: # -0.50
                    data.ctrl[15] -= velocity

                if data.ctrl[16] > target_right_ankle: # -0.25
                    data.ctrl[16] -= velocity

                # print sensor data
                print(data.sensordata)


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


