# Changes

## v.0.1.0 - [21/12/2024]

- gym environment created with darwin op3 model
- action state space normalized
- reward: forward x velocity * 1.50
- terminal state: z < 0.26 or z > 0.32

## v.0.1.1 - [21/12/2024]

- removed 'gymnasiun_env' from registration process
- removed print debug information

## v.0.1.2 - [21/12/2024]

- fixed bug in the velocity calculation

## v.0.1.3 - [21/12/2024]

- changed IMU position to the head

## v.0.1.4 - [21/12/2024]

- calculating velocity using trapezoidal rule
- changed forward default factor to 0.5
- included distance_traveled in the reward function

## v.0.1.5 - [21/12/2024]

- calculating velocity by position difference before and after step
- changed forward default factor to 1.5

## v.0.1.6 - [22/12/2024]

- changed distance traveled reward to be the total distance from the origin without factor

## v.0.1.7 - [22/12/2024]

- readjusting limits and ranges of the model to accomplish a better human simulation
- changed distance traveled reward to be just the distance in the x axis

## v.0.1.8 - [22/12/2024]

- normalizing actions from 0 to 1

## v.0.1.9 - [22/12/2024]

- removed print debug information

## v.0.1.10 - [22/12/2024]

- changed camera track body id

## v.0.1.11 - [22/12/2024]

- removed print debug information

## v.0.1.12 - [22/12/2024]

- limiting observation space state to -20 and 20
- changing normalization of action space state to -1 and 1

## v.0.1.13 - [23/12/2024]

- changing inital pose

## v.0.1.14 - [23/12/2024]

- added distance_traveled reward
- added turn_cost penalty

## v.0.1.15 - [23/12/2024]

- disabled turn_cost penalty

## v.0.1.16 - [23/12/2024]

- changing control mode to force (torque) instead of position

## v.0.1.17 - [23/12/2024]

- added turn_cost penalty

## v.0.1.18 - [23/12/2024]

- added turn_cost penalty only for y axis and with a different factor (0.1)

## v.0.1.19 - [23/12/2024]

- changed turn_cost default factor to 0.05

## v.0.1.20 - [23/12/2024]

- changed ankle limitation to 0.52 radians (30°)
- changed z max limit to 0.31

## v.0.1.21 - [24/12/2024]

- included orientation penalty
- changed distance_traveled to the differente between current and last step on x axis

## v.0.1.22 - [24/12/2024]

- changed turn_cost_weight to 0.01
- changed orientation_cost_weight to 0.01

## v.0.1.23 - [24/12/2024]

- changed turn_cost_weight to 0.03
- changed orientation_cost_weight to 0.03

## v.0.1.24 - [26/12/2024]

- added com_inertia = self.data.cinert[1:].flatten()
- and com_velocity = self.data.cvel[1:].flatten()
- and actuator_forces = self.data.qfrc_actuator[6:].flatten()
- to observation state

- changed default forward_reward_weight to 0
- changed default ctrl_cost_weight to 0
- changed default turn_cost_weight to 0
- changed default orientation_cost_weight to 0

## v.0.1.25 - [26/12/2024]

- removed factor from distance_traveled reward

## v.0.1.26 - [26/12/2024]

- changed distance_traveled reward: if greather then zero = 1 else = 0

## v.0.1.27 - [26/12/2024]

- changed initial pose to only move arms

## v.0.1.28 - [26/12/2024]

- changed distance_traveled reward: distance from origin

## v.0.1.29 - [26/12/2024]

- changed distance_traveled reward: distance from origin * 100
