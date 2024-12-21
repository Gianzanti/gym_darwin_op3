import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
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

    # set default episode_len for truncate episodes
    def __init__(
        self,         
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.50,
        ctrl_cost_weight: float = 0.05,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.260, 0.32),
        reset_noise_scale: float = 1e-2,
        **kwargs):

        utils.EzPickle.__init__(
            self, 
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            **kwargs
        )
        self._forward_reward_weight: float = forward_reward_weight
        self._ctrl_cost_weight: float = ctrl_cost_weight
        self._terminate_when_unhealthy: bool = terminate_when_unhealthy
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._reset_noise_scale: float = reset_noise_scale

        self.accel = np.zeros(3)
        self.vel = np.zeros(3)

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
        print(f"Observation size: {obs_size}")

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Define action space as a symmetric and normalized box
        self.action_space = Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)
      
    def calculate_velocity_from_imu(self):
        """
        Calculates linear velocity from IMU data.

        Args:
            imu_data: A list or array of IMU data points. Each point should be a list/tuple 
                    containing at least linear acceleration (ax, ay, az) in m/s^2.
            dt: Time difference between consecutive IMU readings in seconds.

        Returns:
            A list or array of linear velocities (vx, vy, vz) in m/s.
        """
        prev_velocity = self.vel  # Initial velocity assumed to be zero
        self.accel = np.array(self.data.sensordata[0:3])  # Extract acceleration data
        self.vel = prev_velocity + self.accel * self.dt


    # determine the reward depending on observation or other properties of the simulation
    def step(self, normalized_action):
        # Normalize action to [-pi, pi] range
        action = normalized_action * np.pi 
        print(f"Normalized action: {action}")

        self.do_simulation(action, self.frame_skip)

        self.calculate_velocity_from_imu()
        x_velocity, y_velocity = self.vel[0], self.vel[1]

        observation = self._get_obs()
        reward, reward_info = self._get_rew()
        terminated = (not self.is_healthy)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
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
        return self._forward_reward_weight * self.vel[0]

    def _get_rew(self):
        forward_reward = self.forward_reward()
        reward = forward_reward

        reward_info = {
            "forward_reward": forward_reward,
        }

        return reward, reward_info

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        position = self.data.qpos[2:].flatten()
        velocity = self.data.qvel[2:].flatten()
        imu = self.data.sensordata.flatten()
        return np.concatenate((position,velocity,imu))
    
    def _get_reset_info(self):
        self.calculate_velocity_from_imu()
        x_velocity, y_velocity = self.vel[0], self.vel[1]

        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }        