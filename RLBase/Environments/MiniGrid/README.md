# MiniGrid

Key files:
- `GetEnvironment.py`: holds `MINIGRID_ENV_LST`, the `get_env` vectorizer, and a small CLI demo (`python -m RLBase.Environments.MiniGrid.GetEnvironment`).
- `Wrappers.py`: MiniGrid-specific wrappers and the `WRAPPING_TO_WRAPPER` mapping.
- `CustomEnvironments/`: drop custom MiniGrid env modules here so they can be registered on import.

What `get_env` returns:
- `get_env(env_name, num_envs, max_steps=500, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)` builds a `SyncVectorEnv` with `num_envs` copies of `env_name` (must be in `MINIGRID_ENV_LST`), applies wrappers in the order of `wrapping_lst`, and passes per-wrapper kwargs from `wrapping_params` (defaults are empty).

Adding a custom environment:
- Create a module in `CustomEnvironments/` that defines your env (subclass `MiniGridEnv`) and registers an id via `gymnasium.envs.registration.register`.
- Import the module in `CustomEnvironments/__init__.py` so the registration side effect runs when `RLBase.Environments.MiniGrid` is imported.
- Add the new env id to `MINIGRID_ENV_LST` in `GetEnvironment.py` so `get_env` will allow it.
- (Optional) Expose extra params via `env_params` when calling `get_env`.

Adding a wrapper:
- Implement the wrapper class in `Wrappers.py` (e.g., subclass `gymnasium.Wrapper` / `ObservationWrapper`).
- Register it in `WRAPPING_TO_WRAPPER` with a short string key; that key is what you pass in `wrapping_lst`.
- `get_env` applies wrappers sequentially in the order provided; matching kwargs can be supplied via `wrapping_params` (use `{}` if none).

Quick use:
- Code: `envs = get_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, wrapping_lst=["FullyObs"])` (or use the unified dispatcher `from RLBase.Environments import get_env`)
- CLI smoke test: `python -m RLBase.Environments.MiniGrid.GetEnvironment --env_name MiniGrid-DoorKey-5x5-v0 --num_envs 2 --demo_steps 3`
