# Atari

What lives here:
- `GetEnvironment.py`: exports `ATARI_ENV_LST` and `get_env`; CLI demo via `python -m RLBase.Environments.Atari.GetEnvironment`.
- `Wrappers.py`: Atari wrappers plus the `WRAPPING_TO_WRAPPER` registry (currently only `Identity`).
- `CustomEnvironments/`: drop custom Atari env modules here and import them in `__init__.py` so they register on import.

`get_env` behavior:
- Signature: `get_env(env_name, num_envs, max_steps=None, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`.
- Asserts `env_name` is in `ATARI_ENV_LST`, builds a `SyncVectorEnv` with `gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)`, then applies wrappers in `wrapping_lst` (kwargs pulled from the same index in `wrapping_params`, default `{}`).
- `max_steps` defaults to the gym default when `None`; `render_mode` is omitted when `None`.

Add an Atari id:
1) Implement/register the env (ALE ROM, custom logic, etc.) in `CustomEnvironments/` using `gymnasium.envs.registration.register`.
2) Import the module inside `CustomEnvironments/__init__.py` so registration runs.
3) Append the id to `ATARI_ENV_LST` in `GetEnvironment.py`.
4) Surface any ALE kwargs (e.g., `frameskip`, `repeat_action_probability`) via `env_params` when calling `get_env`.

Add a wrapper:
- Implement the wrapper class in `Wrappers.py` and register it in `WRAPPING_TO_WRAPPER` with the string you want to use in `wrapping_lst`; optional kwargs go in `wrapping_params` at the same position.
- Wrappers run in list order; keep observation/action space definitions in sync if you modify them.

Quick use:
- Code: `envs = get_env("ALE/Pong-v5", num_envs=2, wrapping_lst=["Identity"])` (or call the unified dispatcher `from RLBase.Environments import get_env`).
- CLI: `python -m RLBase.Environments.Atari.GetEnvironment --env ALE/Pong-v5 --num_envs 2 --demo_steps 3`
