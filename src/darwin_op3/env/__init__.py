import importlib.metadata

from gymnasium.envs.registration import find_highest_version, register

env_name = "DarwinOp3"
env_version = 0
env = f"{env_name}-v{env_version}"

env_id = find_highest_version(ns=None, name=env_name)

if env_id is None:
    # Register this module as a gym environment. Once registered, the id is usable in gym.make().
    register(
        id=env,
        entry_point="darwin_op3.env.darwin_env:DarwinEnv",
        nondeterministic=True,
    )
    print(f"Registered environment {env}")

print(f"DarwinOp3 Env version: {importlib.metadata.version('darwin-op3')}")