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
