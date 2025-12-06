# Evaluate

Utilities for inspecting experiment outputs saved by `OnlineTrainer` (see `all_metrics.pkl` and per-run pickles in an `exp_dir`).

What evaluations are supported:
- Per-experiment analysis (`SingleExpAnalyzer`): summarize returns/lengths, plot curves vs episodes or env steps, visualize option usage (from `agent_logs`), save seeds, and render GIFs from stored frames/actions/options.
- Cross-experiment comparisons (`MultiExpAnalyzer` helpers): overlay plots for multiple experiments (`plot_experiments`) and compare option usage across experiments (`plot_option_usage`). `gather_experiments` can load multiple folders by name pattern into a single metrics list.

Expected saved data:
- `all_metrics.pkl`: list of runs; each run is a list of episode dicts containing at least `ep_return` and `ep_length`. Optional fields if trainer enabled them: `frames` (rendered frames or ANSI), `actions`, `transitions` (when `dump_transitions=True`), `agent_logs` (per-step logs such as `OptionUsageLog`, `NumOptions`, `OptionClass`, `OptionIndex`), `agent_seed`, `env_seed`, and `episode_index`.
- Per-run pickles `metrics_run{idx}.pkl` (same structure as above).
- The analyzers accept either the metrics object directly or an `exp_dir` path containing `all_metrics.pkl`.

How to use the analyzers:
- Single experiment:
  ```python
  from RLBase.Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
  ana = SingleExpAnalyzer(exp_path="path/to/exp_dir")  # or metrics=loaded_obj
  ana.print_summary()
  ana.plot_combined(save_dir="plots", plt_configs=("r_e","r_s","s_e"), window_size=10)
  ana.plot_option_class_usage(save_dir="plots", option_classes=[MyOptionClass], x_type="s")
  ana.save_seeds("plots")
  ana.generate_video(run_number=1, episode_number=1, video_type="gif", fps=5)
  ```
- Multiple experiments:
  ```python
  from RLBase.Evaluate.MultiExpAnalyzer import plot_experiments, plot_option_usage
  agent_dict = {"agentA": "exp/A", "agentB": "exp/B"}  # values: exp_dir paths or metrics lists
  plot_experiments(agent_dict, save_dir="plots", name="ret_vs_steps", plt_configs=("r_s","r_e"))
  plot_option_usage(agent_dict, save_dir="plots", option_classes=[MyOptionClass], x_type="s")
  ```
- Bulk loading:
  ```python
  from RLBase.Evaluate.MultiExpAnalyzer import gather_experiments
  metrics = gather_experiments("experiments", name_string_conditions=["PPO"], name_string_anti_conditions=["debug"])
  ```
