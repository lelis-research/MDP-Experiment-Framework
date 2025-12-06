# RLBase Codebase — End-to-End Usage

This repository wires environments, agents, buffers, networks, trainers, and analyzers into runnable scripts for training, sweeping, testing, and visualization. See `RLBase/README.md` for component internals; this file focuses on how to use the full stack.

## Installation
- **Conda** (Python 3.10):  
  ```bash
  conda create -n rlbase python=3.10 -y
  conda activate rlbase
  pip install -r requirements.txt
  python -m pip install -e .
  ```
  Run a script, e.g. `python train.py --help`.

- **Docker/Apptainer**:  
  Build: `docker build -t rlbase .` (or use `rlbase-amd64.tar` with Apptainer: `apptainer build rlbase-amd64.sif rlbase-amd64.tar`).  
  Run: `docker run --rm -v $PWD:/workspace -w /workspace rlbase python train.py --agent PPO --env MiniGrid-Empty-5x5-v0 --total_steps 1000`.  
  On clusters, see `Scripts/*.sh` for SLURM + Apptainer examples.

## Training (`train.py`)
Train an agent with a config module (`Configs/config_agents_base.py`) and override hyperparameters via `--info` JSON.
Example 1 (tabular Q-learning, fixed wrappers, step budget):
```bash
python train.py \
  --config config_agents_base \
  --agent QLearning \
  --env MiniGrid-Empty-5x5-v0 \
  --env_wrapping '["FullyObs","FixedSeed"]' \
  --wrapping_params '[{},{"seed":5}]' \
  --total_steps 20000 \
  --num_runs 3 \
  --num_envs 1 \
  --episode_max_steps 100 \
  --info '{"step_size":0.2,"epsilon_start":1.0,"epsilon_end":0.05,"epsilon_decay_steps":5000}'
```
Example 2 (PPO on Atari with custom networks and checkpoints):
```bash
python train.py \
  --agent PPO \
  --env "ALE/Pong-v5" \
  --num_envs 8 \
  --total_steps 2_000_000 \
  --checkpoint_freq 10_000 \
  --info '{
    "actor_network":"mlp1",
    "critic_network":"mlp1",
    "rollout_steps":256,
    "mini_batch_size":64,
    "num_epochs":4,
    "actor_step_size":0.0003,
    "critic_step_size":0.0003
  }'
```
Outputs go to `Runs/Train/<env>/<wrappers>/<agent>/<name_tag_seed[...]>/` with metrics (`all_metrics.pkl`), TensorBoard/JSONL logs, seeds, and checkpoints.

## Sweeps (`sweep.py`)
Grid-search hyperparameters via SLURM array index `--idx`. Provide a base `--info` and `--hp_search_space` JSON.
Example:
```bash
python sweep.py \
  --idx 0 \
  --agent A2C \
  --env MiniGrid-SimpleCrossingS9N1-v0 \
  --total_steps 500000 \
  --num_runs 2 \
  --hp_search_space '{"actor_step_size":[0.001,0.0003],"critic_step_size":[0.001,0.0003],"entropy_coef":[0.01,0.02]}' \
  --info '{"gamma":0.99,"lamda":0.95,"rollout_steps":256,"update_type":"sync"}'
```
Trials are stored under `Runs/Sweep/<env>/<wrappers>/<agent>/<seed...>/trial_<idx>/`.

## Sweep Analysis (`analyze_sweep.py`)
Evaluate sweep results and plot sensitivities.
Example (analyze last 20% of steps, compute per-step AUC):
```bash
python analyze_sweep.py  # edit exp_dir/ratio/auc_type in-file or run as module
```
Key utilities inside:
- `check_incomplete_runs(exp_dir)` to see missing trials.
- `find_best_hyperparameters(exp_dir, ratio, auc_type)` prints best trial and saves `best_trial_summary-<ratio>.json`.
- `print_info_for_best_trial` prints the `info` dict to reuse in `train.py`.
- `plot_all_param_sensitivities(exp_dir, ratio, out_dir, auc_type)` generates config-colored scatter plots for each swept param.

## Testing & Visualization (`test.py`)
Load trained agents (auto-detects `*_agent.t` under a train directory), run greedy episodes, and render GIFs.
Example:
```bash
python test.py \
  --exp_dir Runs/Train/MiniGrid-Empty-5x5-v0_/FullyObs/QLearning/_seed[123123] \
  --num_runs 3 \
  --render_mode rgb_array_list
```
You can override env/wrappers/params if you want to test in a new setting (`--env`, `--env_wrapping`, etc.). GIFs/plots go to `Runs/Test/.../RunX_Best_agent/`.

## Results Visualization (`visualize_results.py`)
Compares multiple experiments with `plot_experiments`/`plot_option_usage`. Edit the file to point `agent_dict` to experiment folders (or use `gather_experiments` with name filters), then run:
```bash
python visualize_results.py
```
Adjust `plt_configs` to plot returns, lengths, option usage, etc. (`r_s`, `r_e`, `s_e`, `ou_s`, `no_s`).

## Env Sanity Check (`visualize_env.py`)
Render a single frame with chosen wrappers/params:
```bash
python visualize_env.py \
  --env MiniGrid-Empty-5x5-v0 \
  --env_wrapping '["FullyObs","FixedSeed"]' \
  --wrapping_params '[{},{"seed":42}]' \
  --name_tag "demo"
```
Saves PNG to `Runs/Figures/`.

## Compute Canada / SLURM Scripts (`Scripts/`)
Wrapper scripts for Apptainer + SLURM:
- `train_script.sh`, `sweep_script.sh`, `test_script.sh`, `visualize_env_script.sh` show how to set env vars, bind the container (`rlbase-amd64.sif`), and pass arguments to the Python entry points. Edit hyperparameters, arrays, and paths; submit with `sbatch Scripts/train_script.sh`.

## Tips
- Configs live in `Configs/config_agents_base.py`; override per-run via `--info` JSON.
- All runs save `args.yaml`, `config.py`, and metrics for reproducibility.
- Use `tensorboard --logdir Runs/` for live monitoring.
