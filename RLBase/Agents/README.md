# Agents

Agents choose actions, learn from env feedback, and expose metrics for logging/analysis. `OnlineTrainer` calls `agent.act(obs, greedy=not train)` each step and `agent.update(next_obs, reward, terminated, truncated, call_back)` during training. If an agent defines `log()`, the trainer records its per-step dict in episode metrics for later visualization.

Base interface (inherit from `BaseAgent`/`BasePolicy`):
- Required methods: `act(observation, greedy=False)` (return list/array of actions, length = `num_envs`), `update(observation, reward, terminated, truncated, call_back=None)` (training step), `reset(seed)` (seed policy/feature extractor), `set_hp(hp)`, `set_num_env(num_envs)`, `save`/`load`. Policies implement `select_action(state, greedy=False)` where `state` is the batched output of the feature extractor.
- Inputs/outputs: observations are batched (vector env). Feature extractors return dict tensors keyed for networks. Actions must align with env action_space. `call_back` is a logging hook; `log()` (optional) should return a small dict of scalars per step.
- To build a new agent: subclass `BaseAgent`, set `SUPPORTED_ACTION_SPACES`, instantiate a feature extractor and policy, implement `act`/`update`, optionally `log`/`save`/`load`. Keep RNG seeding via `reset`. For new policies, subclass `BasePolicy`.

Sub-packages
------------
**RandomAgent** (`RandomAgent/Random.py`)
- Stateless random policy over `Discrete`, `MultiDiscrete`, or `Box`. No learning; `act` samples independently per env.

**HumanAgent** (`HumanAgent/`)
- `HumanAgent`: interactive control mixing primitive actions and options. Extends action space to primitives + options; keeps a running option until termination. Prints observation summaries; `update` just clears option on done.
- `ContinualHumanAgent`: wraps a placeholder option learner to add options over time; otherwise inherits Human behavior.
- Learning: none in `HumanAgent`; `ContinualHumanAgent` could plug a learner in `option_learner.learn()` when `counter > 100`.

**TabularAgent** (`TabularAgent/`)
- Uses tabular feature extractor (hashable tuples) and discrete actions.
- `QLearningAgent`: epsilon-greedy `QLearningPolicy` with n-step returns. `update` builds per-env rollout buffers; on n-step/full episode computes discounted returns and Q-learning targets `r + γ^n max_a' Q(s',a')`. Logs TD error/epsilon/q stats via callback.
- `OptionQLearningAgent`: extends action space with options (SMDP). Tracks running option per env, accumulates discounted reward and duration, performs SMDP Q-update when option ends (`Q(s,o) ← Q + α(td_target - Q)` with discount `γ^τ`). Primitive steps still use 1-step updates. Can expose option-usage logs via `log()`.

**DeepAgent/ValueBased** (`DeepAgent/ValueBased/`)
- `DQNAgent`/`DQNPolicy`: supports `Discrete` actions, feature dict into value network (`hp.value_network`). Epsilon-greedy from online net; replay buffer with n-step targets; optional Double DQN (`flag_double_dqn_target`). Loss = MSE (or SmoothL1) between Q(s,a) and `r + γ^n max/target`. Target network sync every `target_update_freq`. Logs loss, epsilon, Q stats, TD error, grad norm.
- `OptionDQNAgent`: extends DQN with options. Maintains running option per env, accumulates SMDP returns (`reward sum`, discount multiplier, steps). Adds option transitions to rollout buffer and replay; updates like DQN with effective discount `γ^τ` and action id offset for options. Logs option usage via callback keys `train/option_usage_env_*`.

**DeepAgent/PolicyGradient** (`DeepAgent/PolicyGradient/`)
- Shared: actor/critic networks from configs (`hp.actor_network`, `hp.critic_network`); supports `Discrete` and `Box`. Uses GAE helpers.
- `A2CPolicy`/`A2CAgent`: on-policy n-step rollouts. Losses: policy loss from log-prob × advantage, value MSE, entropy bonus. Uses GAE for returns/advantages. Optimizers Adam for actor/critic.
- `PPOPolicy`/`PPOAgent`: clipped surrogate objective; mini-batch epochs over rollout. Optionally anneals step sizes and clip ranges. Logs clip fraction, losses, entropy, value stats, grad norms via callback.
- `OptionA2C` / `OptionPPO`: extend actor-critic with options/SMDP updates (similar bookkeeping as option Q/DQN). Accumulate discounted returns across option duration; use option termination to trigger updates; propagate option usage via callbacks if needed.

Logs and analyzers
------------------
- Implement `log(self)` to return a dict of scalar metrics per step (e.g., option usage flags, current epsilon, number of options). The trainer will append this to `agent_logs` in metrics; the analyzers can plot keys like `OptionUsageLog`, `NumOptions`, or custom counters. Use `call_back` inside `update` for high-frequency training metrics (losses, gradients) routed to TensorBoard/JSONL.
