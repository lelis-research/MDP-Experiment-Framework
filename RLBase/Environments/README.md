# Environments

Environment families live under this folder (e.g., `MiniGrid`, `Atari`, `Classic`, `MiniHack`, `Mujoco`). Each family ships a `GetEnvironment.py`, `Wrappers.py`, optional `CustomEnvironments/`, and a README with family-specific notes.

Unified loader (`RLBase/Environments/GetEnvironment.py`):
- `get_env(env_name, num_envs, max_steps=None, render_mode=None, env_params=None, wrapping_lst=None, wrapping_params=None)` finds the right family from `ENV_SOURCES`, builds `num_envs` copies via the family `get_env`, and returns a `SyncVectorEnv`.
- `env_params` is forwarded to `gym.make`; `max_steps` and `render_mode` are only passed when provided. Wrappers are applied in order based on the family’s `WRAPPING_TO_WRAPPER` mapping.
- Quick call: `from RLBase.Environments import get_env; envs = get_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, wrapping_lst=["FullyObs"])`
- CLI smoke test: `python -m RLBase.Environments.GetEnvironment --env MiniGrid-DoorKey-5x5-v0 --num_envs 2 --demo_steps 3`

Add a new environment family:
1) Create `RLBase/Environments/<FamilyName>/` containing `__init__.py`, `GetEnvironment.py` (define `<FAMILY>_ENV_LST` and a `get_env` that asserts membership, builds a `SyncVectorEnv`, and applies wrappers), `Wrappers.py` (wrapper classes + `WRAPPING_TO_WRAPPER`), optional `CustomEnvironments/`, and a README.
2) Register it in `RLBase/Environments/GetEnvironment.py` by importing the family `get_env` and list, then adding them to `ENV_SOURCES`. (Optional) re-export from `RLBase/Environments/__init__.py`.
3) (Optional) Add a `__main__` demo in the family `GetEnvironment.py` for quick smoke testing.

Add a new environment id inside an existing family:
- Implement or import the env and register it via `gymnasium.envs.registration.register` in a module under `<Family>/CustomEnvironments/`.
- Import that module inside `<Family>/CustomEnvironments/__init__.py` so registration runs on family import.
- Append the id to `<FAMILY>_ENV_LST` in `<Family>/GetEnvironment.py`; adjust the family default `max_steps` there if you need a different cap.
- Call `get_env(env_name, num_envs, ...)` either directly from the family or through the unified dispatcher.

Create and use wrappers:
- Define wrappers in `<Family>/Wrappers.py` (subclass `gymnasium.Wrapper` / `ObservationWrapper` / `ActionWrapper` / `RewardWrapper` as needed).
- Register them in `WRAPPING_TO_WRAPPER` with the string key that callers will pass in `wrapping_lst`; optional kwargs go in the same slot of `wrapping_params`.
- Wrappers are applied sequentially in the order given. If a wrapper changes observation/action spaces, update the spaces inside the wrapper.

Registering environments:
- Use `register(id=..., entry_point=..., max_episode_steps=..., kwargs=...)` in a module under `CustomEnvironments/`.
- Import that module in `CustomEnvironments/__init__.py` to ensure the register side-effect runs.
- Add the new id to the family list and, if it is a brand-new family, wire it into `ENV_SOURCES` so the unified loader can find it.
