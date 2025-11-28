# MiniGrid

Files:
- `GetEnvironment.py`: `MINIGRID_ENV_LST`, `make_minigrid_env(full_env_name, cfg=None, env_config=None, render_mode=None)`, `make_minigrid_sf_env`, `make_minigrid_sf_factory`, `get_env` (alias).
- `Wrappers.py`: `WRAPPING_TO_WRAPPER` mapping for MiniGrid wrappers.
- `CustomEnvironments/`: import any custom MiniGrid envs here so they are registered before use.

Available env ids: standard MiniGrid set (DoorKey, Empty, MultiRoom, Lava, etc.) plus `BigCurriculumEnv-v0`. Add new ids to `MINIGRID_ENV_LST`.

Wrappers (selected):
- `ViewSize`, `FullyObs`, `SymbolicObs`, `ImgObs`, `RGBImgObs`, `RGBImgPartialObs`
- `StepReward`, `CompactAction`, `FlattenOnehotObj` (legacy flat one-hot), `OnehotObservation` (channel one-hot), `FixedSeed`, `FixedRandomDistractor`
- `FrameStack`, `DropMission`, `AgentPos`, `RecordReward`

How to add a MiniGrid env:
1) Ensure the env id is registered (either via `minigrid` import or by defining it in `CustomEnvironments`).
2) Add the id to `MINIGRID_ENV_LST`.
3) If you need new preprocessing, implement a wrapper in `Wrappers.py` and add it to `WRAPPING_TO_WRAPPER`.
4) `register_sample_factory_envs()` will register it with Sample Factory automatically.
