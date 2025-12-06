# MiniGrid

What lives here:
- `GetEnvironment.py`: exports `MINIGRID_ENV_LST` and `get_env`; CLI demo via `python -m RLBase.Environments.MiniGrid.GetEnvironment`.
- `Wrappers.py`: MiniGrid wrappers plus the `WRAPPING_TO_WRAPPER` registry.
- `CustomEnvironments/`: custom tasks go here; see `CustomEnvironments/MazeRooms.py` for a large registered example.

`get_env` behavior:
- Signature: `get_env(env_name, num_envs, max_steps=500, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`.
- Asserts `env_name` is in `MINIGRID_ENV_LST`, builds a `SyncVectorEnv` with `gym.make(env_name, max_steps=max_steps, render_mode=render_mode, **env_params)`, then applies wrappers in `wrapping_lst` (kwargs pulled from the same index in `wrapping_params`, default `{}`).
- Defaults: `max_steps=500` unless overridden; `render_mode` omitted when `None`.

Add a MiniGrid id:
1) Implement/register the env (subclass `MiniGridEnv`) in `CustomEnvironments/` with `gymnasium.envs.registration.register`.
2) Import that module in `CustomEnvironments/__init__.py` so registration runs (see `MazeRooms` import).
3) Append the id to `MINIGRID_ENV_LST` in `GetEnvironment.py`.
4) Surface any extra kwargs (e.g., `agent_view_size`, rewards) via `env_params` when calling `get_env`.

Add a wrapper:
- Implement it in `Wrappers.py` and register in `WRAPPING_TO_WRAPPER` with the string key you will pass in `wrapping_lst`; kwargs go into `wrapping_params` at the same position.
- Available wrappers: `ViewSize` (changes view field), `FullyObs` (full-grid observations), `DropMission` (removes `mission` key), `OneHotImageDir` (one-hot encodes image + direction), and `FixedSeed` (forces deterministic resets).
- Wrappers run in list order; update observation/action spaces inside wrappers if you change them.

Quick use:
- Code: `envs = get_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, wrapping_lst=["FullyObs"])` (or via the unified dispatcher `from RLBase.Environments import get_env`).
- CLI: `python -m RLBase.Environments.MiniGrid.GetEnvironment --env MiniGrid-DoorKey-5x5-v0 --num_envs 2 --demo_steps 3`
