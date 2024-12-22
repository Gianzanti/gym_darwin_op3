import unittest

import numpy as np


class TestRobotisEnv(unittest.TestCase):
    def test_checkenv(self):
        import gymnasium as gym
        from stable_baselines3.common.env_checker import check_env
        env = gym.make('DarwinOp3-v0', render_mode="human", width=1920, height=1080)
        check_env(env.unwrapped)
        
    def test_rewards(self):
        import gymnasium as gym
        env = gym.make('DarwinOp3-v0', render_mode="human", width=1920, height=1080)

        observation, info = env.reset()
        # print(f"Observation space: {env.observation_space}")
        # print(f"Action space: {env.action_space}")
        # print(f"Info: {info}")
        # print(f"Action sample: {env.action_space.sample()}")

        episode_over = False
        counter = 0
        action_one = 0.00
        # increment = 0.01
        increment = 0.00
        step = increment
        action = np.zeros(env.action_space.shape[0])
        while not episode_over:
            if counter % 100 == 0:
                action[0] = action_one
                action[1] = action_one
                action[2] = action_one
                action[3] = action_one
                action[4] = action_one
                action[5] = action_one

            observation, reward, terminated, truncated, info = env.step(action)
            # print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            # print(f"Terminated: {terminated}")
            # print(f"Truncated: {truncated}")
            # print(f"Info: {info}")
            

            episode_over = counter > 1000
            if (action_one > 3.14):
                step = -increment
            if (action_one < -3.14):
                step = increment
            action_one += step
            counter += 1

        env.close()

        print("Environment check successful!")

    def test_imu(self):
        import gymnasium as gym
        env = gym.make('DarwinOp3-v0', render_mode="human", width=1024, height=768)

        observation, info = env.reset()
        # print(f"Observation space: {env.observation_space}")
        # print(f"Action space: {env.action_space}")
        # print(f"Info: {info}")
        # print(f"Action sample: {env.action_space.sample()}")

        episode_over = False
        counter = 0
        action_one = 0.00
        increment = 0.01
        step = increment
        action = np.zeros(env.action_space.shape[0])
        while not episode_over:
            if counter % 100 == 0:
                action[0] = action_one
        #         action[1] = action_one
        #         action[2] = action_one
        #         action[3] = action_one
        #         action[4] = action_one
        #         action[5] = action_one

            observation, reward, terminated, truncated, info = env.step(action)
        # print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")
        # print(f"Info: {info}")
            counter += 1
            print(f"c: {counter}")
            episode_over = counter > 10000
                
            if (action_one > 1):
                step = -increment
            if (action_one < -1):
                step = increment

            action_one += step
                

        #         # print(f"Info: {info}")
        #     counter += 1
        #     # episode_over = False

        env.close()

        print("Environment check successful!")