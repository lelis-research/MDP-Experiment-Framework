# Classic

What lives here:
- `GetEnvironment.py`: exports `CLASSIC_ENV_LST` and `get_env`; CLI demo via `python -m RLBase.Environments.Classic.GetEnvironment`.
- `Wrappers.py`: common Gymnasium wrappers registered in `WRAPPING_TO_WRAPPER` (frame stacking, normalization, clipping, rescaling, reward recording, etc.).
- `CustomEnvironments/`: add modules here for custom classic-control/toy-text tasks and import them in `__init__.py` so they register on import.

`get_env` behavior:
- Signature: `get_env(env_name, num_envs, max_steps=500, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`.
- Asserts `env_name` is in `CLASSIC_ENV_LST`, builds a `SyncVectorEnv` with `gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)`, then applies wrappers in `wrapping_lst` (kwargs pulled from the same index in `wrapping_params`, default `{}`).
- `CLASSIC_ENV_LST` currently covers control and toy-text ids such as `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar[-Continuous]-v0`, and `CliffWalking-v1`/`FrozenLake-v1`/`Taxi-v3`/`Blackjack-v1`.

Add a Classic id:
1) If it is a built-in Gymnasium id, add it to `CLASSIC_ENV_LST` and ensure any dependencies are installed.
2) For custom envs, register with `gymnasium.envs.registration.register` inside `CustomEnvironments/`, import that module in `CustomEnvironments/__init__.py`, and append the id to `CLASSIC_ENV_LST`.
3) Adjust the `max_steps` default in `get_env` if the new task needs a different cap, or pass per-call overrides via `max_steps` / `env_params`.

Add a wrapper:
- Implement it in `Wrappers.py` (subclass `gymnasium.Wrapper`, `ObservationWrapper`, `ActionWrapper`, or `RewardWrapper` as appropriate) and register the class in `WRAPPING_TO_WRAPPER` with the string key you want to use in `wrapping_lst`.
- Notable existing entries: `FrameStackObservation`, `TransformObservation`, `NormalizeObservation`, `RescaleObservation`, `ClipAction`, `TransformAction`, `RescaleAction`, `ClipReward`, `NormalizeReward`, and `RecordActualReward` (adds `info["actual_reward"]` before other reward wrappers).
- Wrappers run in list order; keep observation/action spaces consistent when you transform them.

Quick use:
- Code: `envs = get_env("CartPole-v1", num_envs=4, wrapping_lst=["NormalizeObservation", "NormalizeReward"])`.
- CLI: `python -m RLBase.Environments.Classic.GetEnvironment --env CartPole-v1 --num_envs 2 --demo_steps 3`
