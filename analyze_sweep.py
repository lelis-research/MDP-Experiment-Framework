"""

After running the sweep.py, this script provides utilities to:
  1) Report any trials that have not finished (missing METRICS_FILE or AGENT_FILE).
  2) Find and display the best hyperparameter combination (overall and per-run averages).

"""
import os
import pickle
import numpy as np
import yaml
import json

SYMLINK_PREFIX = 'trial_'  # prefix for trial directories
METRICS_FILE = 'all_metrics.pkl'
AGENT_FILE = 'agent.txt'
ARGS_FILE = 'args.yaml'


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
    num_complete = 0
    incomplete = []
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        files_ok = os.path.isfile(os.path.join(trial_dir, METRICS_FILE)) and \
                   os.path.isfile(os.path.join(trial_dir, AGENT_FILE))
        if not files_ok:
            incomplete.append(trial)
        else:
            num_complete += 1
    return incomplete, num_complete


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


def load_args_yaml(trial_dir):
    """
    Load args.yaml (if present) from a trial directory and return the parsed dict.
    Returns None if yaml is unavailable or file missing/invalid.
    """
    args_path = os.path.join(trial_dir, ARGS_FILE)
    if not os.path.isfile(args_path):
        return None
    if yaml is None:
        return None
    try:
        with open(args_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def get_info_dict_from_trial(trial_dir):
    """
    Return the 'info' dictionary from args.yaml for a given trial directory.
    If missing, returns {}.
    """
    args_data = load_args_yaml(trial_dir)
    if not isinstance(args_data, dict):
        return {}
    info = args_data.get('info', {})
    return info if isinstance(info, dict) else {}


def print_info_for_best_trial(exp_dir, ratio, sort_keys=True, indent=2):
    """
    Find best trial, load its args.yaml, and print the INFO dict
    as a JSON object suitable for copy-paste into train_script.sh.
    """
    best = find_best_hyperparameters(exp_dir, ratio)
    if best is None:
        print("No completed trials found to analyze.")
        return
    overall_avg, run_avgs, agent_str, trial_name = best
    trial_dir = os.path.join(exp_dir, trial_name)
    info_dict = get_info_dict_from_trial(trial_dir)

    print("\nINFO dict for train_script.sh (copy & paste):")
    if not info_dict:
        print("  (args.yaml missing or has no 'info' block)")
        return

    # Pretty JSON (multi-line) for readability
    pretty = json.dumps(info_dict, indent=indent, sort_keys=sort_keys)
    print(pretty)

    # Also provide a compact one-liner (handy for single-line Bash assignments)
    # compact = json.dumps(info_dict, separators=(',', ':'), sort_keys=sort_keys)
    # print("\nOne-liner JSON:")
    # print(compact)
    
    

def main(exp_dir, ratio):
    # 1) Check incomplete trials
    incomplete, num_complete = check_incomplete_runs(exp_dir)
    if incomplete:
        print("Incomplete trials (missing files):")
        for t in incomplete:
            print(f"  {t}")
    print(f"{num_complete} trials have metrics and agent info.")

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
        
    # 4) Print INFO dict for copy-paste
    print_info_for_best_trial(exp_dir, ratio)


if __name__ == '__main__':
    # --- Configuration ---
    exp_dir = "Runs/Sweep/MiniGrid-DoorKey-5x5-v0_/SymbolicObs_FixedSeed(seed-10)/OptionQLearning/_seed[1]"
    ratio   = 0.5
    # ---------------------

    main(exp_dir, ratio)
