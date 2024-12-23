import math
import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 2.5,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -5.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class DarwinEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,         
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.5,
        ctrl_cost_weight: float = 5e-2,
        turn_cost_weight: float = 3e-2,
        orientation_cost_weight: float = 5e-2,
        healthy_z_range: Tuple[float, float] = (0.260, 0.310),
        reset_noise_scale: float = 1e-2,
        **kwargs):

        utils.EzPickle.__init__(
            self, 
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            turn_cost_weight,
            orientation_cost_weight,
            healthy_z_range,
            reset_noise_scale,
            **kwargs
        )
        self._forward_reward_weight: float = forward_reward_weight
        self._ctrl_cost_weight: float = ctrl_cost_weight
        self._turn_cost_weight: float = turn_cost_weight
        self._orientation_cost_weight: float = orientation_cost_weight
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._reset_noise_scale: float = reset_noise_scale

        self.velocity = np.zeros(2)
        self.x_pos = 0

        xml_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos[2:].size + self.data.qvel[2:].size + self.data.sensordata.size
        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        # )
        self.observation_space = Box(
            low=-20, high=20, shape=(obs_size,), dtype=np.float64
        )

        self.greaterX = 0
        self.greaterY = 0
        self.greaterZ = 0

        # Define a dict of ranges / min-max values for each action
        # self.joint_ranges = {
        #     "l_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "l_sho_roll": {"min": -0.60, "max": 1.90, "range": 2.50},
        #     "l_el": {"min": -3.00, "max": 0.50, "range": 3.50},
        #     "r_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "r_sho_roll": {"min": -1.90, "max": 0.60, "range": 2.50},
        #     "r_el": {"min": -0.50, "max": 3.00, "range": 3.50},
        #     "l_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_pitch": {"min": -2.00, "max": 1.0, "range": 3.0},
        #     "l_knee": {"min": -0.08, "max": 3.00, "range": 3.08},
        #     "l_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "l_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_pitch": {"min": -1.0, "max": 2.00, "range": 3.0},
        #     "r_knee": {"min": -3.00, "max": 0.08, "range": 3.08},
        #     "r_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        # }
        # self.joint_ranges = {
        #     "l_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "l_sho_roll": {"min": -1.90, "max": 1.90, "range": 3.80},
        #     "l_el": {"min": -3.00, "max": 3.0, "range": 6.00},
        #     "r_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "r_sho_roll": {"min": -1.90, "max": 1.90, "range": 3.80},
        #     "r_el": {"min": -3.00, "max": 3.00, "range": 6.00},
        #     "l_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_pitch": {"min": -2.00, "max": 2.0, "range": 4.0},
        #     "l_knee": {"min": -3.00, "max": 3.00, "range": 6.00},
        #     "l_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "l_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_pitch": {"min": -2.0, "max": 2.00, "range": 4.0},
        #     "r_knee": {"min": -3.00, "max": 3.00, "range": 6.00},
        #     "r_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        # }
        self.action_space = Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)
        

    # determine the reward depending on observation or other properties of the simulation
    def step(self, normalized_action):
        xy_position_before = mass_center(self.model, self.data)
        
        # Normalize action to its range
        # action = np.zeros(self.action_space.shape[0])
        # for i, joint_range in enumerate(self.joint_ranges.values()):
        # #     action[i] = joint_range["min"] + normalized_action[i] * joint_range["range"]
        # for i, joint_range in enumerate(self.joint_ranges.values()):
        #     action[i] = normalized_action[i] * (joint_range["range"]/2)

        # action = normalized_action * np.pi 
        self.do_simulation(normalized_action * 5, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        self.velocity = (xy_position_after - xy_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew()
        terminated = (not self.is_healthy)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "orientation": self.data.qpos[3],
            "x_velocity": self.velocity[0],
            "y_velocity": self.velocity[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    def control_cost(self):
        return -(self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl)))

    def distance_cost(self):
        target_position = np.array(self._target_position) # ball target position 
        distance_to_target = np.linalg.norm(self.data.qpos[0:2] - target_position, ord=2)
        return -distance_to_target
    
    def calculate_feet_center_of_mass(self):
        if self._zmp_weight == 0:
            return 0
        center_x = (self.data.site('r_foot').xpos[0] + self.data.site('l_foot').xpos[0]) / 2
        center_y = (self.data.site('r_foot').xpos[1] + self.data.site('l_foot').xpos[1]) / 2
        foot_mass_center = np.array([center_x, center_y])
        distance2 = np.linalg.norm(foot_mass_center - self.data.site('torso').xpos[0:2], ord=2)
        return distance2 * self._zmp_weight

    def forward_reward(self):
        return self._forward_reward_weight * self.velocity[0]

    def distance_traveled(self):
        last_position = self.x_pos
        distance_traveled = last_position - self.data.qpos[0]
        self.x_pos = self.data.qpos[0]
        return self._forward_reward_weight * distance_traveled

    def cost_y_axis_angular_velocity(self):
        y_ang_vel = math.pow(self.data.sensordata[4], 2)
        return self._turn_cost_weight * y_ang_vel

    def cost_orientation(self):
        orientation = math.pow(1 - self.data.qpos[3], 2)
        return self._orientation_cost_weight * orientation

    def _get_rew(self):
        forward_reward = self.forward_reward()
        # distance_traveled = self.data.qpos[0]
        distance_traveled = self.distance_traveled()
        y_vel_ang = self.cost_y_axis_angular_velocity()
        x_orientation = self.cost_orientation()
        # turn_cost = 0

        reward = forward_reward + distance_traveled - y_vel_ang - x_orientation

        reward_info = {
            "forward_reward": forward_reward,
            "distance_traveled": distance_traveled,
            "y_vel_ang": y_vel_ang,
            "x_orientation": x_orientation
        }

        return reward, reward_info


    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.init_qpos[8] = 1.30
        self.init_qpos[11] = -1.30
        self.init_qpos[15] = -0.45
        self.init_qpos[16] = 0.70
        self.init_qpos[17] = 0.25
        self.init_qpos[21] = 0.45
        self.init_qpos[22] = -0.70
        self.init_qpos[23] = -0.25

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        # qpos[8] = 1.30
        # qpos[11] = -1.30
        # qpos[15] = -0.45
        # qpos[16] = 0.70
        # qpos[17] = 0.25
        # qpos[21] = 0.45
        # qpos[22] = -0.70
        # qpos[23] = -0.25


        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        position = self.data.qpos[2:]
        velocity = self.data.qvel[2:]
        imu = self.data.sensordata
        return np.concatenate((position,velocity,imu))
    
    def _get_reset_info(self):
        # self.calculate_velocity_from_imu()

        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "orientation": self.data.qpos[3],
            "x_velocity": 0,
            "y_velocity": 0,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }        