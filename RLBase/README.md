# RLBase

Lightweight RL toolkit with pluggable components. The flow is: **environment → feature extractor → agent (policy + networks + buffers) → trainer → metrics/analyzers**. Everything is configured via small Python dicts (networks, agents, envs) and a registry for discoverability.

Components
- **Environments (`Environments/`)**: Families (MiniGrid, Atari, MuJoCo, Classic, MiniHack). `get_env` builds vectorized envs, applies wrappers, and supports custom registrations. See each family README for adding envs/wrappers and registering IDs.
- **FeatureExtractors (`FeatureExtractors/`)**: Map observations to model-ready features. Function-approximation extractors return dicts of tensors keyed for networks; tabular returns hashable states. Implement by subclassing `BaseFeature`, filling `features_dict`, `__call__`, and registering with `register_feature_extractor`.
- **Networks (`Networks/`)**: Graph-based builder (`NetworkGen`) driven by layer configs; `prepare_network_config` auto-fills shapes. Initializers (`LayerInit.py`) and presets (`Presets.py`) are provided. Agents reference network configs via their hyperparameters.
- **Buffers (`Buffers/`)**: Storage for experience—uniform replay, prioritized replay (PER), and HER. Subclass `BaseBuffer`/`RandomGenerator` to add new sampling schemes; expose `add/sample` and any update hooks.
- **Agents (`Agents/`)**: Policies + learning logic. Base interfaces in `Agents/Base.py` define `act`, `update`, `reset`, `save`, etc. Subfolders: Random, Human, Tabular (QLearning/OptionQLearning), Deep Value-Based (DQN/OptionDQN), Deep Policy-Gradient (A2C/PPO/option variants). Agents consume feature extractors, networks, and buffers; they emit logs via callbacks and optional `log()` for analyzer use.
- **Trainers (`Trainers/`)**: `OnlineTrainer` runs vector envs, calls `agent.act/update`, handles multi-run execution, logging (TensorBoard/JSONL), checkpointing, and metrics pickles. Config/args are saved to `exp_dir`.
- **Evaluate (`Evaluate/`)**: `SingleExpAnalyzer` and `MultiExpAnalyzer` plot returns/lengths, option usage, and render videos from stored frames/actions. Load metrics from `all_metrics.pkl` or pass metrics objects directly.
- **Options (`Options/`)**: Utilities to define/load/save options used by option-enabled agents (Option DQN/A2C/PPO/Q-learning).
- **Configs (`Configs/`)**: Example config loaders for agents/options; shows how to assemble hyperparameters and presets.
- **Registry (`registry.py`)**: Lightweight name-to-class mappings for agents, policies, feature extractors. Decorators `@register_agent`, `@register_policy`, `@register_feature_extractor` make components discoverable.
- **Utilities (`utils.py`, `Agents/Utils`)**: RNG mixin `RandomGenerator`, observation/state helpers, GAE/n-step calculators, layer init helpers, etc.

How things connect
1) **Environment**: Built via `RLBase.Environments.get_env(env_name, num_envs, wrapping_lst, ...)`, yields a `SyncVectorEnv`.
2) **Agent construction**: Given `env.action_space`, `env.observation_space`, hyperparameters, feature extractor class, and device. Agent instantiates its feature extractor, networks, and buffers internally.
3) **Training loop**: `OnlineTrainer.multi_run(...)` creates fresh env/agent per run, seeds them, then iterates `act → env.step → update`. Trainer logs episode returns/lengths and forwards agent callback logs to TB/JSONL; optional `agent.log()` entries are stored in episode metrics.
4) **Saving/loading**: Trainer writes args/config, metrics pickles, and agent checkpoints. Agents/policies/feature extractors implement `save/load` for reproducibility.
5) **Analysis**: Analyzers read `all_metrics.pkl` to plot performance, option usage, and render videos; they use any keys you logged via callbacks or `agent.log()`.

Contributing / extending
- **New env**: Register a Gymnasium env in the appropriate family’s `CustomEnvironments`, add its ID to `<FAMILY>_ENV_LST`, wire wrappers if needed.
- **New feature extractor**: Subclass `BaseFeature`, return dict tensors, register it; update agents to accept it.
- **New network functionality**: Add layer/init types in `NetworkFactory`/`LayerInit`, create presets in `Presets`.
- **New buffer**: Subclass `BaseBuffer` or `RandomGenerator`, implement `sample` and any update hooks.
- **New agent**: Subclass `BaseAgent`/`BasePolicy`, set `SUPPORTED_ACTION_SPACES`, build networks/buffers in `__init__`, implement `act`/`update`, and optionally `log` for analyzer plots. Register with `@register_agent`.
- **Logging/analysis**: Use `call_back` in `update` to emit training scalars (losses, eps, grad norms) and `log()` for per-step episode metadata (e.g., option usage) to visualize with analyzers.

Getting started
- Pick an env via `Environments`, a feature extractor, and an agent class from the registry. Prepare network configs/presets and agent hyperparameters (see `Configs/`).
- Instantiate `OnlineTrainer(env_factory, agent_factory, exp_dir=...)`, then run `multi_run(num_runs, num_episodes=... or total_steps=...)`.
- Inspect results with `Evaluate` plots and logs in `exp_dir`. Modify configs/agents to explore new architectures or algorithms.
