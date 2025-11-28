# Environments (Sample Factory style)

This folder is now Sample Factory–first. Every domain (Atari, MiniGrid, MiniHack, Mujoco) follows the same structure:

- `GetEnvironment.py`
  - `*_ENV_LST`: list of supported env ids.
  - `make_<domain>_env(full_env_name, cfg=None, env_config=None, render_mode=None)`: builds a single gym/gymnasium env, applies wrappers, and optionally seeds it.
  - `make_<domain>_sf_env` (alias) and `make_<domain>_sf_factory`: helpers for Sample Factory registration.
  - `get_env`: alias to `make_<domain>_env` to keep imports working.
- `Wrappers.py`
  - `WRAPPING_TO_WRAPPER`: mapping from string keys to wrapper classes to be used by `make_<domain>_env`.

Common config keys read by all `make_*_env`:
- `env_params` (or `env_kwargs` in `env_config`): kwargs forwarded to `gym.make`.
- `env_wrappers` / `env_wrapper_params` (or legacy `wrapping_lst` / `wrapping_params`): parallel lists of wrapper names + kwargs.
- `env_max_steps` / `max_steps`: episode length override.
- `render_mode`
- `seed` (optional reset seed)

Registration with Sample Factory:
- Call `register_sample_factory_envs()` from `RLBase.Environments.sf_registration` once in your runner; it will register all env ids using the `make_*_sf_factory` helpers.

Adding a new environment to an existing domain:
1. Add the env id to the domain’s `*_ENV_LST`.
2. Ensure the env is registered with gym/gymnasium (or import the custom env in `CustomEnvironments` for MiniGrid/Mujoco if needed).
3. If custom wrappers are required, add them to the domain’s `Wrappers.py` and to `WRAPPING_TO_WRAPPER`.
4. Nothing else is needed for SF; `register_sample_factory_envs` will pick up the new id.

Adding a new domain folder:
1. Create `NewDomain/GetEnvironment.py` and `NewDomain/Wrappers.py` following the same function names and mapping pattern.
2. Export the new functions/ids in `NewDomain/__init__.py`.
3. Add the ids and factories to `sf_registration.py` and to the top-level `GetEnvironment.py`/`ENV_LST`/`make_env` dispatcher.
4. Add a short `README.md` in the new domain describing its env ids and wrappers.
