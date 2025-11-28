# MuJoCo

Key files:
- `GetEnvironment.py`: builds MuJoCo vector envs, exposes `MUJOCO_ENV_LST`, `get_env`, and a CLI demo (`python -m RLBase.Environments.Mujoco.GetEnvironment`).
- `Wrappers.py`: MuJoCo wrappers plus the `WRAPPING_TO_WRAPPER` mapping used by `get_env`.
- `CustomEnvironments/`: place custom MuJoCo env modules here so they can register on import.

What `get_env` returns:
- `get_env(env_name, num_envs, max_steps=1000, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)` builds a `SyncVectorEnv` with `num_envs` copies of `env_name` (must be in `MUJOCO_ENV_LST`), applies wrappers in `wrapping_lst`, and passes per-wrapper kwargs from `wrapping_params` (defaults are empty). `max_steps` is forwarded as `max_episode_steps` unless you override it in `env_params`; `render_mode` is added only if provided.

Adding a custom environment:
- Create a module in `CustomEnvironments/` that defines your env (e.g., via a MuJoCo XML or subclassing `gym.Env`) and register an id via `gymnasium.envs.registration.register`.
- Import that module in `CustomEnvironments/__init__.py` so registration runs when `RLBase.Environments.Mujoco` is imported.
- Add the new id to `MUJOCO_ENV_LST` in `GetEnvironment.py` so `get_env` allows it.
- Surface extra knobs through `env_params` when calling `get_env`.

Adding a wrapper:
- Implement the wrapper in `Wrappers.py` (subclass `gymnasium.Wrapper` / `ObservationWrapper`, etc.).
- Register it in `WRAPPING_TO_WRAPPER` with a string key; pass that key in `wrapping_lst` and kwargs in `wrapping_params`.
- Wrappers apply sequentially in the order listed.

Quick use:
- Code: `envs = get_env("HalfCheetah-v4", num_envs=2, wrapping_lst=["RecordEpisodeStatistics"])` (or use `from RLBase.Environments import get_env` to go through the unified dispatcher)
- CLI smoke test: `python -m RLBase.Environments.Mujoco.GetEnvironment --env_name HalfCheetah-v4 --num_envs 2 --demo_steps 3`
