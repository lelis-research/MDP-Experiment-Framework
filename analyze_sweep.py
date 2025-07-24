"""

After running the sweep.py, this script provides utilities to:
  1) Report any trials that have not finished (missing METRICS_FILE or AGENT_FILE).
  2) Find and display the best hyperparameter combination (overall and per-run averages).

"""
import os
import pickle
import numpy as np

SYMLINK_PREFIX = 'trial_'  # prefix for trial directories
METRICS_FILE = 'all_metrics.pkl'
AGENT_FILE = 'agent.txt'


def compute_run_avgs(metrics, ratio):
    """
    Compute average reward for each run over the last `ratio` fraction of episodes.
    Returns a list of per-run averages.
    """
    run_avgs = []
    for run in metrics:
        returns = [ep.get('ep_return', 0.0) for ep in run]
        n = len(returns)
        if n == 0:
            run_avgs.append(0.0)
            continue
        idx = int(n * ratio)
        idx = max(0, min(idx, n - 1))
        run_avgs.append(float(np.mean(returns[idx:])))
    return run_avgs


def compute_avg_reward(metrics, ratio):
    """
    Compute overall average reward across runs using run-level averages.
    """
    run_avgs = compute_run_avgs(metrics, ratio)
    return float(np.mean(run_avgs)) if run_avgs else 0.0


def find_trials(exp_dir):
    """
    Return a sorted list of trial directory names under exp_dir.
    """
    return sorted(d for d in os.listdir(exp_dir) if d.startswith(SYMLINK_PREFIX))


def check_incomplete_runs(exp_dir):
    """
    Identify trials missing metrics.pkl or agent.txt.
    Returns a list of trial names that are incomplete.
    """
    incomplete = []
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        files_ok = os.path.isfile(os.path.join(trial_dir, METRICS_FILE)) and \
                   os.path.isfile(os.path.join(trial_dir, AGENT_FILE))
        if not files_ok:
            incomplete.append(trial)
    return incomplete


def find_best_hyperparameters(exp_dir, ratio):
    """
    Scan completed trials, compute run-level and overall average rewards,
    and return (trial_name, agent_str, overall_avg, run_avgs).
    """
    results = []  # (overall_avg, run_avgs, agent_str, trial)
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        metrics_path = os.path.join(trial_dir, METRICS_FILE)
        agent_path   = os.path.join(trial_dir, AGENT_FILE)
        if not os.path.isfile(metrics_path) or not os.path.isfile(agent_path):
            continue
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        with open(agent_path, 'r') as f:
            agent_str = f.read().strip()
        run_avgs = compute_run_avgs(metrics, ratio)
        overall_avg = float(np.mean(run_avgs)) if run_avgs else 0.0
        results.append((overall_avg, run_avgs, agent_str, trial))

    if not results:
        return None
    return max(results, key=lambda x: x[0])


def main(exp_dir, ratio):
    # 1) Check incomplete trials
    incomplete = check_incomplete_runs(exp_dir)
    if incomplete:
        print("Incomplete trials (missing files):")
        for t in incomplete:
            print(f"  {t}")
    else:
        print("All trials have metrics and agent info.")

    # 2) Find best hyperparameters among completed
    best = find_best_hyperparameters(exp_dir, ratio)
    if best is None:
        print("No completed trials found to analyze.")
        return
    overall_avg, run_avgs, agent_str, trial_name = best

    print("\nBest hyperparameter combination:")
    print(f"  Trial:          {trial_name}")
    print(f"  Agent details:  {agent_str}")
    print(f"  Overall avg:    {overall_avg:.6f}\n")

    # 3) Per-run averages
    print("Per-run average rewards:")
    for i, r in enumerate(run_avgs, 1):
        print(f"  Run {i}: {r:.6f}")


if __name__ == '__main__':
    # --- Configuration ---
    exp_dir = "Runs/Sweep/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-30_seed-100)/OptionDQN/Transfer_NoDistractor_seed[1]"
    ratio   = 0.5
    # ---------------------

    main(exp_dir, ratio)
