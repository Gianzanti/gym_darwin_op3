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

## v.0.1.30 - [26/12/2024]

- changed distance_traveled reward: distance from origin

## v.0.1.31 - [26/12/2024]

- disabled distance_traveled reward

## v.0.1.32 - [28/12/2024]

- set model options:  

    ```xml
        <option gravity="0 0 -9.81" integrator="RK4" iterations="50" solver="PGS" timestep="0.003" />
    ```

- changed motor default:

    ```xml
        <motor gear="2" ctrllimited="true" ctrlrange="-3 3" />
    ```

- observation:
    qpos: [-1.81733535e-02  2.63665246e-07  2.73120558e-01  9.98971214e-01
            -4.97820770e-06 -4.53487950e-02  4.66168740e-07  1.96844518e-05
            8.78612017e-01 -2.08919183e-01 -1.96818791e-05 -8.78612261e-01
            2.08919245e-01 -2.77229828e-05  6.71360742e-05 -2.02440087e-01
            4.39676006e-01  1.46577957e-01  4.78351161e-04  2.81710759e-05
            -8.33294949e-05  2.02420865e-01 -4.39648438e-01 -1.46569615e-01
            -4.77869095e-04]
    qvel: [-9.73362410e-02  4.83459871e-06 -4.04525944e-02 -3.48260308e-05
    -5.73587196e-01 -4.28049363e-06  2.19773364e-04  2.46113404e+00
    -5.41479827e-01 -2.19791677e-04 -2.46113578e+00  5.41480209e-01
    -1.43023659e-04 -1.63305603e-05 -5.00450293e-01  1.36947432e+00
    2.95779804e-01  1.76572573e-05  1.44258448e-04 -1.58547116e-05
    5.00451803e-01 -1.36947731e+00 -2.95780776e-01  2.09238542e-05]
    Inertia: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00]
    [ 8.44604839e-03  8.20947345e-03  3.37236264e-03  7.09991046e-06
    9.94589747e-04 -6.76007500e-06 -1.66241583e-02  1.09558189e-04
    8.24256256e-02  1.34928000e+00]
    [ 1.21397084e-03  1.20770667e-03  1.20768685e-05 -6.21594491e-09
    1.68643215e-05  3.99150046e-07 -1.25045972e-04 -2.73646117e-06
    8.22378583e-03  5.63346681e-02]
    [ 1.33529360e-02  1.32612207e-02  1.46247866e-04  5.32927045e-08
    -9.83480174e-05  4.70699113e-06  4.94581084e-04 -2.48892969e-05
    6.63064068e-02  3.36114357e-01]
    [ 1.86068838e-04  1.33660249e-04  5.86796754e-05  1.77668947e-06
    2.57088872e-06 -8.39068477e-05 -2.64700904e-05  8.01816101e-04
    1.23516501e-03  1.17600000e-02]
    [ 3.38289975e-03  8.57998974e-04  2.56553376e-03 -5.35715885e-05
    -1.34728278e-05 -1.21465820e-03  3.78838337e-04  2.08613965e-02
    1.11897499e-02  1.77580000e-01]
    [ 1.47331584e-03  1.00564150e-04  1.40085380e-03 -7.92130521e-05
    1.67269019e-05  2.67473031e-04  4.22405581e-04  7.51583168e-03
    -1.33063710e-03  4.12700000e-02]
    [ 1.86227414e-04  1.33663600e-04  5.88348935e-05 -1.77920292e-06
    2.57084303e-06  8.40274054e-05 -2.64693145e-05 -8.02953499e-04
    1.23518104e-03  1.17600000e-02]
    [ 3.38702361e-03  8.58047517e-04  2.56960929e-03  5.36112098e-05
    -1.34750744e-05  1.21579782e-03  3.78858650e-04 -2.08787184e-02
    1.11901655e-02  1.77580000e-01]
    [ 1.47480095e-03  1.00553700e-04  1.40234967e-03  7.92564580e-05
    1.67255934e-05 -2.67578636e-04  4.22412958e-04 -7.51993441e-03
    -1.33048796e-03  4.12700000e-02]
    [ 1.73723600e-05  6.42189083e-06  1.92321533e-05 -3.19025383e-06
    7.73950735e-07  4.52037308e-06  9.12770424e-05  4.12764746e-04
    -1.29336895e-04  1.18100000e-02]
    [ 6.37940613e-04  5.04620078e-04  3.40584116e-04  4.79186617e-05
    -6.99207997e-05  2.85893740e-04 -1.38867545e-03  6.30101661e-03
    -8.11968702e-03  1.78860000e-01]
    [ 1.68227441e-03  1.68956351e-03  3.22822963e-04 -1.46744766e-04
    4.87878839e-04  4.54190816e-04  4.17114900e-03  4.03248469e-03
    -1.29047078e-02  1.15430000e-01]
    [ 1.54876503e-03  1.54677059e-03  1.07197699e-04 -4.56683243e-05
    2.66273400e-04  2.49115846e-04  1.40829219e-03  1.30198918e-03
    -7.68212379e-03  4.01500000e-02]
    [ 1.00292774e-02  9.89443777e-03  3.38696688e-04 -4.40029356e-05
    2.86506946e-04  1.47141596e-03  1.22085357e-03  6.29840238e-03
    -4.17831310e-02  1.78860000e-01]
    [ 5.36353818e-03  5.30890514e-03  3.07526603e-04 -8.40174115e-05
    5.05764241e-04  8.59869079e-04  1.84987597e-03  3.14211658e-03
    -1.89535660e-02  6.93400000e-02]
    [ 1.74540420e-05  6.42171618e-06  1.93140224e-05  3.19930050e-06
    7.73891298e-07 -4.53289523e-06  9.12774421e-05 -4.13934218e-04
    -1.29328638e-04  1.18100000e-02]
    [ 6.39185190e-04  5.04608618e-04  3.41839970e-04 -4.80568291e-05
    -6.99195523e-05 -2.86697483e-04 -1.38866929e-03 -6.31881711e-03
    -8.11956179e-03  1.78860000e-01]
    [ 1.68307466e-03  1.68954518e-03  3.23614685e-04  1.47154492e-04
    4.87857133e-04 -4.55476583e-04  4.17097468e-03 -4.04399161e-03
    -1.29046823e-02  1.15430000e-01]
    [ 1.54902570e-03  1.54676608e-03  1.07454076e-04  4.58071321e-05
    2.66261698e-04 -2.49884037e-04  1.40823018e-03 -1.30600387e-03
    -7.68212336e-03  4.01500000e-02]
    [ 1.00305453e-02  9.89443895e-03  3.39958156e-04  4.41182735e-05
    2.86461672e-04 -1.47560338e-03  1.22065955e-03 -6.31632585e-03
    -4.17831392e-02  1.78860000e-01]
    [ 5.36416971e-03  5.30890166e-03  3.08153728e-04  8.41995404e-05
    5.05744087e-04 -8.61770190e-04  1.84980215e-03 -3.14907194e-03
    -1.89535668e-02  6.93400000e-02]]
    Velocity: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00]
    [-3.44834299e-05 -5.73600759e-01 -4.49280141e-06 -9.91159108e-02
    4.98819098e-06 -4.53610796e-02]
    [-3.44834299e-05 -5.73600759e-01 -4.49280141e-06 -9.91159108e-02
    4.98819098e-06 -4.53610796e-02]
    [-3.44834299e-05 -5.73600759e-01 -4.49280141e-06 -9.91159108e-02
    4.98819098e-06 -4.53610796e-02]
    [-3.44835362e-05 -5.73380956e-01 -4.49499664e-06 -9.91395071e-02
    4.98817417e-06 -4.53616208e-02]
    [-2.45105407e+00 -5.73384367e-01 -2.22945886e-01 -1.18858690e-01
    -2.39052531e-01  1.71434904e-01]
    [-2.99030836e+00 -5.73385118e-01 -2.71995719e-01 -1.26023329e-01
    -2.53929029e-01  2.50203238e-01]
    [-3.44835362e-05 -5.73380961e-01 -4.49499659e-06 -9.91395065e-02
    4.98817417e-06 -4.53616207e-02]
    [ 2.45098688e+00 -5.73377549e-01  2.22937063e-01 -1.18880965e-01
    2.39066994e-01  1.71672479e-01]
    [ 2.99024155e+00 -5.73376799e-01  2.71986933e-01 -1.26050470e-01
    2.53945062e-01  2.50493849e-01]
    [-4.74526009e-05 -5.73600757e-01  1.38059272e-04 -9.91109285e-02
    3.80373600e-06 -4.53606263e-02]
    [-2.79181986e-05 -5.73600757e-01  1.39836482e-04 -9.91108664e-02
    3.16886078e-06 -4.53613091e-02]
    [-1.69035523e-05 -1.07404788e+00  1.79553952e-04 -1.14860940e-01
    2.37444740e-06 -5.10031689e-02]
    [-4.70454756e-05  2.95441285e-01  7.08658560e-05  7.26525670e-02
    1.11866106e-05  8.02946283e-03]
    [-4.05354289e-05 -3.40716773e-04  9.43402898e-05 -3.40254589e-05
    8.95188504e-06  2.93336197e-05]
    [-2.11665916e-05 -3.40716347e-04  9.43416600e-05 -3.40253063e-05
    4.19202263e-06  2.86570075e-05]
    [-2.13905562e-05 -5.73600760e-01 -1.48404567e-04 -9.91108667e-02
    6.18395758e-06 -4.53606207e-02]
    [-8.51165780e-06 -5.73600760e-01 -1.47232865e-04 -9.91109078e-02
    5.76539764e-06 -4.53601693e-02]
    [-1.85314462e-05 -1.07405129e+00 -1.85045048e-04 -1.14860802e-01
    6.50701431e-06 -5.10020507e-02]
    [ 8.88787019e-06  2.95439843e-01 -8.15713848e-05  7.26530269e-02
    -1.70738016e-06  8.02784356e-03]
    [ 2.96584887e-06 -3.42719590e-04 -1.03919617e-04 -3.37543841e-05
    3.52356299e-07  2.80401419e-05]
    [ 2.24166521e-05 -3.42719980e-04 -1.03918240e-04 -3.37545281e-05
    -4.42765002e-06  2.87215678e-05]]

## v.0.1.33 - [28/12/2024]

- changed timestep to 0.001

## v.0.1.34 - [28/12/2024]

- changed render_fps metadata to 200

## v.0.1.35 - [28/12/2024]

- changed render_fps metadata to 100
- changed model options:  

    ```xml
        <option gravity="0 0 -9.81" integrator="RK4" iterations="50" solver="PGS"/>
    ```

## v.0.1.36 - [28/12/2024]

- changed model options:  

    ```xml
        <option gravity="0 0 -9.81" integrator="RK4"/>
    ```

## v.0.1.37 - [28/12/2024]

- changed health limits to (0.270, 0.310)
- changed orientation_cost_weight to 5e-2

## v.0.1.38 - [28/12/2024]

- changed turn_cost_weight to 3e-2

## v.0.1.39 - [29/12/2024]

- removed qfrc_actuator as observation state
- changed turn_cost_weight to 5e-2

## v.0.1.40 - [29/12/2024]

- changed turn_cost_weight to 1

## v.0.1.41 - [29/12/2024]

- changed orientation_cost_weight to 1

## v.0.1.42 - [29/12/2024]

- turn_cost_weight: float = 1.25, #5e-2,
- orientation_cost_weight: float = 1.25, #5e-2,
- healthy_z_range: Tuple[float, float] = (0.270, 0.290),
- enabled distance traveled
- changed calculation for orientation penalty

## v.0.1.47 - [30/12/2024]

- removed turn_cost and orientation_cost
- added rotation_penalty

## v.0.1.48 - [30/12/2024]

- adjusting rotation_penalty

## v.0.2.23 - [01/04/2025]

- updated libraries: "gymnasium>=1.1.1", "mujoco>=3.3.0", "stable-baselines3=2.6.0"

## v.0.2.25 - [18/04/2025]

- adding distance reward height

## v.0.2.26 - [18/04/2025]

- adding distance reward height

## v.0.2.27 - [20/04/2025]

- changing reward function to calculate y deviations
