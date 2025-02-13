from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        super().__init__(verbose)
        # self.x_pos = (0,0)
        # self.y_pos = (0,0)
        # self.z_pos = (0,0)
        # self.x_vel = (0,0)
        # self.y_vel = (0,0)
        # self.forward_reward = (0,0)
        # self.distance_traveled = (0,0)
        # self.rotation_penalty = (0,0)
        # self.control_cost = (0,0)
        print("001 - Tensorboard Callback Initialized")
        self.episode_positions = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print("002 - Training Started")

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("003 - Rollout Started")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % 1000 == 0:
            print("004 - Step: ", self.num_timesteps)
        info = self.locals['infos'][0]
        # self.logger.record('info/pos_z', info['z_position'])

        self.episode_positions.append(info['z_position'])

        # calculate the mean of the values
        # self.x_pos = ((self.x_pos[0] + info['x_position']) / 2, max(self.x_pos[1], info['x_position']))
        # self.y_pos = ((self.y_pos[0] + info['y_position']) / 2, max(self.y_pos[1], info['y_position']))
        # self.z_pos = ((self.z_pos[0] + info['z_position']) / 2, max(self.z_pos[1], info['z_position']))
        # self.x_vel = ((self.x_vel[0] + info['x_velocity']) / 2, max(self.x_vel[1], info['x_velocity']))
        # self.y_vel = ((self.y_vel[0] + info['y_velocity']) / 2, max(self.y_vel[1], info['y_velocity']))
        # self.forward_reward = ((self.forward_reward[0] + info['forward_reward']) / 2, max(self.forward_reward[1], info['forward_reward']))
        # self.distance_traveled = ((self.distance_traveled[0] + info['distance_traveled']) / 2, max(self.distance_traveled[1], info['distance_traveled']))
        # self.rotation_penalty = ((self.rotation_penalty[0] + info['rotation_penalty']) / 2, max(self.rotation_penalty[1], info['rotation_penalty']))
        # self.control_cost = ((self.control_cost[0] + info['control_cost']) / 2, max(self.control_cost[1], info['control_cost']))


        # log the mean values
        # self.logger.record('mean_info/pos_x', self.x_pos[0])
        # self.logger.record('mean_info/pos_y', self.y_pos[0])
        # self.logger.record('mean_info/pos_z', self.z_pos[0])
        # self.logger.record('mean_info/vel_x', self.x_vel[0])
        # self.logger.record('mean_info/vel_y', self.y_vel[0])
        # self.logger.record('mean_reward/forward_reward', self.forward_reward[0])
        # self.logger.record('mean_reward/distance_traveled', self.distance_traveled[0])
        # self.logger.record('mean_reward/rotation_penalty', self.rotation_penalty[0])
        # self.logger.record('mean_reward/control_cost', self.control_cost[0])

        # log the max values
        # self.logger.record('max_info/pos_x', self.x_pos[1])
        # self.logger.record('max_info/pos_y', self.y_pos[1])
        # self.logger.record('max_info/pos_z', self.z_pos[1])
        # self.logger.record('max_info/vel_x', self.x_vel[1])
        # self.logger.record('max_info/vel_y', self.y_vel[1])
        # self.logger.record('max_reward/forward_reward', self.forward_reward[1])
        # self.logger.record('max_reward/distance_traveled', self.distance_traveled[1])
        # self.logger.record('max_reward/rotation_penalty', self.rotation_penalty[1])
        # self.logger.record('max_reward/control_cost', self.control_cost[1])
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print("005 - Rollout Ended")
        if self.episode_positions:
            self.logger.record('mean_episode/pos_z', sum(self.episode_positions) / len(self.episode_positions))
            self.logger.record('max_episode/pos_z', max(self.episode_positions))
            self.logger.record('min_episode/pos_z', min(self.episode_positions))
            self.episode_positions = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("006 - Training Ended")

