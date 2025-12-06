# MiniHack

What lives here:
- `GetEnvironment.py`: exports `MINIHACK_ENV_LST` (currently `["MiniHack-River-v0"]`) and `get_env`; CLI demo via `python -m RLBase.Environments.MiniHack.GetEnvironment`.
- `Wrappers.py`: placeholder registry `WRAPPING_TO_WRAPPER` (add MiniHack-specific wrappers here).
- `CustomEnvironments/`: add custom MiniHack/NLE tasks here and import them in `__init__.py` so they register on import.

`get_env` behavior:
- Signature: `get_env(env_name, num_envs, max_steps=500, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`.
- Asserts `env_name` is in `MINIHACK_ENV_LST`, builds a `SyncVectorEnv` with `gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)`, then applies wrappers in `wrapping_lst` (kwargs pulled from the same index in `wrapping_params`, default `{}`).

Wire MiniHack into the unified loader:
- In `RLBase/Environments/GetEnvironment.py`, import `get_env`/`MINIHACK_ENV_LST` from this folder and add them to `ENV_SOURCES` (imports are currently commented out).

Add a MiniHack id:
1) Implement/register the env with `gymnasium.envs.registration.register` inside `CustomEnvironments/`.
2) Import that module in `CustomEnvironments/__init__.py` so registration runs.
3) Append the id to `MINIHACK_ENV_LST` in `GetEnvironment.py`.
4) Pass MiniHack kwargs (e.g., `observation_keys`, `actions`) through `env_params` when calling `get_env`.

Add a wrapper:
- Define it in `Wrappers.py` (subclass `gymnasium.Wrapper` / `ObservationWrapper` / `ActionWrapper` / `RewardWrapper`) and register it in `WRAPPING_TO_WRAPPER` with the string key you will pass in `wrapping_lst`; kwargs go into `wrapping_params` at the same index.
- Wrappers run in list order; adjust observation/action spaces inside wrappers if you modify them.

Quick use:
- Code: `envs = get_env("MiniHack-River-v0", num_envs=2)` (import directly from this package or through the unified dispatcher once MiniHack is wired into `ENV_SOURCES`).
- CLI: `python -m RLBase.Environments.MiniHack.GetEnvironment --env MiniHack-River-v0 --num_envs 2 --demo_steps 3`
