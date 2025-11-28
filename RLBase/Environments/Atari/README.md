# Atari

Key files:
- `GetEnvironment.py`: builds Atari vector envs, exposes `ATARI_ENV_LST`, `get_env`, and a CLI demo (`python -m RLBase.Environments.Atari.GetEnvironment`).
- `Wrappers.py`: Atari wrappers plus the `WRAPPING_TO_WRAPPER` mapping used by `get_env`.
- `CustomEnvironments/`: place custom Atari env modules here so they can register on import.

What `get_env` returns:
- `get_env(env_name, num_envs, max_steps=None, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)` builds a `SyncVectorEnv` with `num_envs` copies of `env_name` (must be in `ATARI_ENV_LST`), applies wrappers in `wrapping_lst`, and passes per-wrapper kwargs from `wrapping_params` (defaults are empty). `max_steps` is forwarded as `max_episode_steps` unless you override it in `env_params`; `render_mode` is added only if provided.

Adding a custom environment:
- Create a module in `CustomEnvironments/` that defines your env (e.g., via an ALE ROM wrapper) and register an id via `gymnasium.envs.registration.register`.
- Import that module in `CustomEnvironments/__init__.py` so registration runs when `RLBase.Environments.Atari` is imported.
- Add the new id to `ATARI_ENV_LST` in `GetEnvironment.py` so `get_env` allows it.
- Surface extra knobs through `env_params` when calling `get_env`.

Adding a wrapper:
- Implement the wrapper in `Wrappers.py` (subclass `gymnasium.Wrapper` / `ObservationWrapper`, etc.).
- Register it in `WRAPPING_TO_WRAPPER` with a string key; pass that key in `wrapping_lst` and kwargs in `wrapping_params`.
- Wrappers apply sequentially in the order listed.

Quick use:
- Code: `envs = get_env("ALE/Pong-v5", num_envs=2, wrapping_lst=["RecordEpisodeStatistics"])` (or use `from RLBase.Environments import get_env` for the unified dispatcher)
- CLI smoke test: `python -m RLBase.Environments.Atari.GetEnvironment --env_name ALE/Pong-v5 --num_envs 2 --demo_steps 3`
