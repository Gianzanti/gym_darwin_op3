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
        env = gym.make('DarwinOp3-v0', render_mode="human", width=1024, height=768)

        observation, info = env.reset()
        # print(f"Observation space: {env.observation_space}")
        # print(f"Action space: {env.action_space}")
        # print(f"Info: {info}")
        # print(f"Action sample: {env.action_space.sample()}")

        episode_over = False
        counter = 0
        action = np.zeros(env.action_space.shape[0])

        target_left_shoulder = 0.80
        target_right_shoulder = -0.80

        target_left_hip = -0.45
        target_left_knee = 0.70
        target_left_ankle = 0.25

        target_right_hip = 0.45
        target_right_knee = -0.70
        target_right_ankle = -0.25
        
        velocity = 0.005
        
    #   <key time="1" qpos='0.00 0.00 0.279106 0.99993 0.00 0.00 0.00 0.00 1.30 -0.30 0.00 -1.30 0.30 0.00 0.00 -0.25 0.50 0.25 0.00 0.00 0.00 0.25 -0.50 -0.25 0.00' />

        while not episode_over:
            observation, reward, terminated, truncated, info = env.step(action)

            if observation[6] < target_left_shoulder: # 1.30
                action[1] += velocity * 3
                if action[1] > target_left_shoulder:
                    action[1] = target_left_shoulder

            if observation[9] > target_right_shoulder: # -1.30
                action[4] -= velocity * 3
                if action[4] < target_right_shoulder:
                    action[4] = target_right_shoulder

            if observation[13] > target_left_hip: # -0.25
                action[8] -= velocity
                if action[8] < target_left_hip:
                    action[8] = target_left_hip
            
            if observation[14] < target_left_knee: # 0.50
                action[9] += velocity * 1.5
                if action[9] > target_left_knee:
                    action[9] = target_left_knee

            if observation[15] < target_left_ankle: # 0.25
                action[10] += velocity
                if action[10] > target_left_ankle:
                    action[10] = target_left_ankle

            if observation[19] < target_right_hip: # 0.25
                action[14] += velocity
                if action[14] > target_right_hip:
                    action[14] = target_right_hip
            
            if observation[20] > target_right_knee: # -0.50
                action[15] -= velocity * 1.5
                if action[15] < target_right_knee:
                    action[15] = target_right_knee

            if observation[21] > target_right_ankle: # -0.25
                action[16] -= velocity
                if action[16] < target_right_ankle:
                    action[16] = target_right_ankle


            
            # print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            # print(f"Terminated: {terminated}")
            # print(f"Truncated: {truncated}")
            # print(f"Info: {info}")
            episode_over = counter > 1000
            counter += 1

        env.close()

        print("Environment check successful!")
