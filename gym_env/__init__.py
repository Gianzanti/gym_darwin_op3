from importlib.metadata import version

from gymnasium.envs.registration import find_highest_version, register

env_name = "DarwinOp3"
env = f"{env_name}-v1"
env_id = find_highest_version(ns=None, name=env_name)

if env_id is None:
    # Register this module as a gym environment. Once registered, the id is usable in gym.make().
    register(
        id=env,
        entry_point="gym_env.envs:DarwinEnv",
        nondeterministic=True,
    )
    print(f"Registered environment {env}")

# print(f"DarwinOp3 Env version: {version('darwin-op3')}")
print("DarwinOp3 Env version 0.2.2")