"""

After running the grid sweep, this script reads each trial’s
- metrics.pkl (pickled list of runs with episode dicts)
- agent.txt   (string representation of the agent and its HyperParameters)

It computes the average reward over the last `metric_ratio` fraction of episodes
for each run, then across runs, and reports the best hyperparameter set.

Usage:
    python analyze_sweep.py --runs-dir Runs/Sweep --metric_ratio 0.5
"""
import os
import argparse
import pickle
import numpy as np

SYMLINK_PREFIX = 'trial_'  # prefix for trial directories



def compute_avg_reward(metrics, ratio):
    run_avgs = []
    for run in metrics:
        returns = [ep.get('ep_return', 0.0) for ep in run]
        n = len(returns)
        if n == 0:
            run_avgs.append(0.0)
            continue
        idx = int(n * ratio)
        idx = max(0, min(idx, n - 1))
        run_avgs.append(np.mean(returns[idx:]))
    return float(np.mean(run_avgs)) if run_avgs else 0.0


def main(exp_dir, ratio):

    # Find all trial directories
    trials = sorted(d for d in os.listdir(exp_dir) if d.startswith(SYMLINK_PREFIX))
    if not trials:
        print(f"No trials found under {exp_dir}")
        return

    results = []  # list of (avg_reward, agent_str, trial_name)
    for trial in trials:
        trial_dir = os.path.join(exp_dir, trial)
        metrics_file = os.path.join(trial_dir, 'metrics.pkl')
        agent_file   = os.path.join(trial_dir, 'agent.txt')

        if not os.path.isfile(metrics_file):
            print(f"Warning: missing metrics.pkl in {trial_dir}")
            continue
        if not os.path.isfile(agent_file):
            print(f"Warning: missing agent.txt in {trial_dir}")
            continue

        # Load metrics and agent string
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
        with open(agent_file, 'r') as f:
            agent_str = f.read().strip()

        # Compute average reward
        avg_reward = compute_avg_reward(metrics, ratio)
        results.append((avg_reward, agent_str, trial))

    if not results:
        print("No valid results to analyze.")
        return

    # Get best by max average reward
    best_reward, best_agent, best_trial = max(results, key=lambda x: x[0])

    print("Best hyperparameter combination:")
    print(f"  Trial:          {best_trial}")
    print(f"  Agent details:  {best_agent}")
    print(f"  Avg. reward:    {best_reward:.6f}\n")

    # Optionally show sorted results
    print("All trials sorted by avg reward (desc):")
    for avg, agent, trial in sorted(results, key=lambda x: x[0], reverse=True):
        print(f"  {trial}: reward={avg:.6f} -> {agent}")

if __name__ == '__main__':
    ratio = 0.5
    exp_dir = ""
    main(exp_dir=exp_dir, ratio=ratio)
