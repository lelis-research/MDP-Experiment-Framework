# Trainers

`OnlineTrainer` runs vectorized environments, coordinates agents, and logs metrics. `CallBacks.py` houses logging utilities (TensorBoard, JSONL, no-op).

Create a trainer instance:
- Provide `env` and `agent` either as ready-made instances or factories; factories enable multi-process `num_workers`. Both are normalized to internal factories.
- Common signature: `trainer = OnlineTrainer(env, agent, exp_dir="runs/exp1", train=True, config="config.py", args=cli_args)`.
- `exp_dir` (optional but recommended) stores logs/checkpoints/metrics; `config` (path) is copied there; `args` (Namespace) is serialized to `args.yaml`. `train=False` switches to greedy acting without updates.
- Default callback is `TBCallBack(log_dir=exp_dir, flush_every=1)`; swap to `JsonlCallback`/`EmptyCallBack` by assigning `trainer.call_back`.

How the training loop works:
- Call `multi_run(num_runs, num_episodes=..., total_steps=..., seed_offset=None, dump_metrics=True, checkpoint_freq=None, dump_transitions=False, num_workers=1, tuning_hp=None)`.
- Exactly one of `num_episodes` or `total_steps` must be set. Episode mode uses `_single_run_episodes`; step mode uses `_single_run_steps` and logs episode summaries when a sub-env ends.
- Per run: builds fresh env/agent (factories recommended), seeds each sub-env, sets `agent.set_num_env`, then repeats `act → step → update`. `agent.update` is called every step during training with a callback hook for agent-level metrics.
- Dones are detected via `terminated | truncated`; returns/lengths accumulate per sub-env. Best agent snapshot is tracked by highest episode return and saved when `checkpoint_freq` is set. Progress shown via `tqdm`.
- Rendering: supports `"human"`, `"ansi"`, `"rgb_array"`, `"rgb_array_list"`; frames collected when available.

What gets saved:
- Per-episode metrics (each entry in a run list): `ep_return`, `ep_length`, optional `frames`, optional `transitions` (if `dump_transitions=True`), `actions` (logged by default), `agent_seed`, `episode_index`, `agent_logs` (entries from `agent.log()` each step).
- Pickles: `metrics_run{idx}.pkl` per run and `all_metrics.pkl` aggregated when `dump_metrics=True`.
- Checkpoints: when `checkpoint_freq` is set, saves last/best agent for each run in `exp_dir` (`Run{run}_Last`, `Run{run}_Best_agent.t`). Agent’s own `save` method can write extra files.
- Config artifacts: copies `config.py` (if provided) and writes `args.yaml` (from `args` Namespace) to `exp_dir`.
- Helpers: `OnlineTrainer.load_transitions(exp_dir)`, `.load_args(exp_dir)`, `.load_config(exp_dir)` to reload stored outputs.

Callbacks: purpose and usage:
- `TBCallBack`: buffered/averaged TensorBoard logger; `force=True` writes immediately (used for episode summaries). Default for the trainer.
- `JsonlCallback`: fast newline-delimited JSON logger; good for clusters without TensorBoard.
- `BasicCallBack`: direct TensorBoard scalar writer without buffering.
- `EmptyCallBack`: no-op.
- Swap by setting `trainer.call_back = JsonlCallback(...)` (or similar). Agent updates receive a `call_back` function to emit agent-specific metrics with tags like `agents/run_{i}`; trainer logs episode returns/lengths under `ep_return/run_{i}` and `ep_length/run_{i}`.
