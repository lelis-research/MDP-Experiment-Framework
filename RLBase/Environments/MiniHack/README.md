# MiniHack

Files:
- `GetEnvironment.py`: `MINIHACK_ENV_LST`, `make_minihack_env(full_env_name, cfg=None, env_config=None, render_mode=None)`, `make_minihack_sf_env`, `make_minihack_sf_factory`, `get_env` (alias).
- `Wrappers.py`: `WRAPPING_TO_WRAPPER` mapping for MiniHack wrappers.

Available env ids: `MiniHack-River-v0` (extend as needed).

Wrappers:
- `OneHotChars` (one-hot encodes the `chars` grid)
- `MovementAction` (restricts actions to compass directions)

How to add a MiniHack env:
1) Ensure the env id is registered by `minihack` (importing `minihack` already registers defaults) or your custom registration.
2) Add the id to `MINIHACK_ENV_LST`.
3) Add any new wrappers to `Wrappers.py` and `WRAPPING_TO_WRAPPER` if needed.
4) Use `register_sample_factory_envs()` in your runner to expose it to Sample Factory.
