# Environments

This package groups environment families and a unified loader:
- Families: `MiniGrid`, `Mujoco`, `Atari` (each has its own `GetEnvironment.py`, `Wrappers.py`, `CustomEnvironments/`, `README.md`).
- Unified loader: `RLBase/Environments/GetEnvironment.py` dispatches to the right family via `get_env(env_name, num_envs, max_steps=None, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)`. Accepts any id from the family lists and returns a `SyncVectorEnv`.

Quick unified usage:
- Code: `from RLBase.Environments import get_env; envs = get_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, wrapping_lst=["FullyObs"])`
- CLI smoke test: `python -m RLBase.Environments.GetEnvironment --env MiniGrid-DoorKey-5x5-v0 --num_envs 2 --demo_steps 3`

Family-level details:
- MiniGrid: see `MiniGrid/README.md` for env list, wrappers, and custom env registration.
- Mujoco: see `Mujoco/README.md` for supported ids and wrapper mapping.
- Atari: see `Atari/README.md` for supported ids and wrapper mapping.

Extending:
- Add new env ids to the respective family list and register them (via `gymnasium.envs.registration.register`) in that family's `CustomEnvironments/`.
- Add new wrappers by implementing them in the family `Wrappers.py` and registering in `WRAPPING_TO_WRAPPER`; pass names/params through `wrapping_lst`/`wrapping_params`.
