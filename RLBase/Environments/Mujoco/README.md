# MuJoCo

What lives here:
- `GetEnvironment.py`: exports `MUJOCO_ENV_LST` and `get_env`; CLI demo via `python -m RLBase.Environments.Mujoco.GetEnvironment`. The import of `gymnasium_robotics` ensures standard robotics tasks are registered.
- `Wrappers.py`: MuJoCo wrapper registry `WRAPPING_TO_WRAPPER` (currently only `Identity`).
- `CustomEnvironments/`: add custom MuJoCo/XML tasks here and import them in `__init__.py` so they register on import.

`get_env` behavior:
- Signature: `get_env(env_name, num_envs, max_steps=1000, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`.
- Asserts `env_name` is in `MUJOCO_ENV_LST`, builds a `SyncVectorEnv` with `gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)`, then applies wrappers in `wrapping_lst` (kwargs pulled from the same index in `wrapping_params`, default `{}`).
- `max_steps` defaults to 1000 unless overridden; `render_mode` is omitted when `None`.

Add a MuJoCo id:
1) Implement/register the env (custom XML, robotics variant, etc.) in `CustomEnvironments/` with `gymnasium.envs.registration.register`.
2) Import that module in `CustomEnvironments/__init__.py` so registration runs.
3) Append the id to `MUJOCO_ENV_LST` in `GetEnvironment.py`.
4) Pass MuJoCo kwargs (e.g., XML path, frame-skip) through `env_params` when calling `get_env`.

Add a wrapper:
- Implement it in `Wrappers.py` and register it in `WRAPPING_TO_WRAPPER` with the string key you will pass in `wrapping_lst`; kwargs go into `wrapping_params` at the matching index.
- Wrappers run in list order; adjust observation/action spaces inside wrappers if you modify them.

Quick use:
- Code: `envs = get_env("HalfCheetah-v5", num_envs=2, wrapping_lst=["Identity"])` (or call through the unified dispatcher `from RLBase.Environments import get_env`).
- CLI: `python -m RLBase.Environments.Mujoco.GetEnvironment --env HalfCheetah-v5 --num_envs 2 --demo_steps 3`
