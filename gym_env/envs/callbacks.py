import numpy as np
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
        # print("001 - Tensorboard Callback Initialized")
        # print("001 - Tensorboard Callback Initialized")
        # print(f"Num Envs: {self.training_env.num_envs}")
        # print(self.__dict__)
        self.episode_positions = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("002 - Training Started")
        # print(f"Num Envs: {self.training_env.num_envs}")
        # self.episode_positions = [[]] * self.training_env.num_envs
        self.episode_positions = []


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print("003 - Rollout Started")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        for env_idx in range(self.training_env.num_envs):
            info = self.locals['infos'][env_idx]
            self.episode_positions.append(info['z_position'])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print("005 - Rollout Ended")
        if self.episode_positions:
            z_values = np.array(self.episode_positions) # Convert to NumPy array for calculations
            self.logger.record('mean_episode/pos_z', np.mean(z_values))
            self.logger.record('max_episode/pos_z', np.max(z_values))
            self.logger.record('min_episode/pos_z', np.min(z_values))
            self.episode_positions = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # print("006 - Training Ended")

