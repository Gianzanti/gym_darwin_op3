import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 2.5,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -5.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    com = np.sum(mass * xpos, axis=0) / np.sum(mass)
    return com[0:2].copy()


class DarwinOp3Env(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        pos_deviation_weight: float = 10.0,  # 5e-2,
        lateral_velocity_weight: float = 5.0,  # 5e-2,
        
        ctrl_cost_weight: float = 1e-3,  # 5e-2,
        reach_target_reward: float = 100.0,  # 10000.0,
        target_distance: float = 5.0,  # 5.0
        
        forward_velocity_weight: float = 1.0,  # 2.50,
        keep_alive_reward: float = 1.0,  # 0.1
        healthy_z_range: Tuple[float, float] = (0.265, 0.310),
        motor_max_torque: float = 3.0,  # 3.0,
        reset_noise_scale: float = 1e-2,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            pos_deviation_weight,
            lateral_velocity_weight,
            ctrl_cost_weight,
            reach_target_reward,
            target_distance,
            forward_velocity_weight,
            keep_alive_reward,
            healthy_z_range,
            motor_max_torque,
            reset_noise_scale,
            **kwargs,
        )

        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "mjcf", "scene.xml"
        )

        self._pos_deviation_weight: float = pos_deviation_weight
        self._lateral_velocity_weight: float = lateral_velocity_weight
        self._ctrl_cost_weight: float = ctrl_cost_weight
        self._reach_target_reward: float = reach_target_reward
        self._target_distance: float = target_distance
        self._fw_vel_rew_weight: float = forward_velocity_weight
        self._keep_alive_reward: float = keep_alive_reward
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._motor_max_torque = motor_max_torque
        self._reset_noise_scale: float = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        obs_size = self.data.qpos[2:].size + self.data.qvel[2:].size
        obs_size += self.data.sensordata.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _get_obs(self):
        position = self.data.qpos[2:].flatten()
        velocity = self.data.qvel[2:].flatten()
        gyro = self.data.sensordata[0:3].flatten()
        acc = self.data.sensordata[3:6].flatten()
        mag = self.data.sensordata[6:9].flatten()

        return np.concatenate((position, velocity, gyro, acc, mag))

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

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        return min_z < self.data.qpos[2] < max_z

    def _get_rew(self, velocity, position_before, position_after):
        [x_velocity, y_velocity] = velocity
        health_reward = self._keep_alive_reward * self.is_healthy
        forward_reward = self._fw_vel_rew_weight * x_velocity
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        pos_deviation_cost = self._pos_deviation_weight * (self.data.qpos[1] ** 2)
        lateral_velocity_cost = self._lateral_velocity_weight * (y_velocity ** 2)
        
        reward = (
            health_reward + forward_reward + control_cost - pos_deviation_cost - lateral_velocity_cost
        )

        if self.data.qpos[0] >= self._target_distance:
            health_reward = 0
            forward_reward = 0
            control_cost = 0
            pos_deviation_cost = 0
            lateral_velocity_cost = 0
            reward = self._reach_target_reward

        reward_info = {
            "health_reward": health_reward,
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "pos_deviation_cost": pos_deviation_cost,
            "lateral_velocity_cost": lateral_velocity_cost,
        }

        return reward, reward_info

    def termination(self):
        if not self.is_healthy:
            return True

        if self.data.qpos[0] >= self._target_distance:
            return True

        return False

    def step(self, normalized_action):
        # get the current position of the robot, before action
        position_before = mass_center(self.model, self.data)

        # denormalize the action to the range of the motors
        action = normalized_action * self._motor_max_torque
        self.do_simulation(action, self.frame_skip)
        position_after = mass_center(self.model, self.data)

        velocity = (position_after - position_before) / self.dt
        distance_from_origin = np.linalg.norm(self.data.qpos[0:2], ord=2)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(velocity, position_before, position_after)

        info = {
            "x_position": position_after[0],
            "y_position": position_after[1],
            "z_position": self.data.qpos[2],
            "orientation": self.data.qpos[3],
            "x_velocity": velocity[0],
            "y_velocity": velocity[1],
            "distance_from_origin": distance_from_origin,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, self.termination(), False, info
