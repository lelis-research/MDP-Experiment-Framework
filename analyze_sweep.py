"""
Utilities for analyzing sweeps:
  1) Report trials that have not finished (missing METRICS_FILE or AGENT_FILE).
  2) Find and display the best hyperparameter combo (overall + per-run averages).
  3) Sensitivity plots: for each hyper-param, show all runs as dots, colored by
     the 'other-params' configuration, with a center line across x (mean/median).
"""

import os
import pickle
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from collections import defaultdict

SYMLINK_PREFIX = 'trial_'   # prefix for trial directories
METRICS_FILE   = 'all_metrics.pkl'
AGENT_FILE     = 'agent.txt'
ARGS_FILE      = 'args.yaml'


# ----------------------------
# Basic sweep utilities
# ----------------------------
def compute_run_avgs_by_episodes_AUC(metrics, ratio: float):
    """
    Tail AUC on the episode axis (no safety checks).
    Assumes:
      - each episode has 'ep_return'
      - each run has at least 1 episode
      - 0 < ratio <= 1
    Returns per-run averages over the last `ratio` fraction of episodes.
    """
    out = []
    for run in metrics:
        returns = [float(ep['ep_return']) for ep in run]
        n = len(returns)

        window = ratio * n                    # tail width on episode index axis
        k_full = int(window)                  # number of full episodes
        frac   = window - k_full              # fractional part in (0,1) or 0

        total = 0.0
        if k_full > 0:
            total += sum(returns[n - k_full:])        # full episodes
        if frac > 0.0:
            total += returns[n - k_full - 1] * frac   # partial boundary episode

        out.append(total / window)
    return out


def compute_run_avgs_by_steps_AUC(
    metrics,
    ratio_steps: float = None,    # e.g., 0.2 -> last 20% of total steps
    last_steps: int = None,       # or a fixed window size in steps (takes precedence)
    normalize: str = "per_episode",  # "per_step" or "per_episode"
):
    """
    Compute AUC of the last steps across runs.
    Assumes each episode dict has 'ep_return' and 'ep_length'.
    No safety checks — expects all data to be valid.
    """
    run_avgs = []
    for run in metrics:
        R = [float(ep["ep_return"]) for ep in run]
        L = [int(ep["ep_length"]) for ep in run]

        total_steps = sum(L)
        window_steps = (
            last_steps
            if last_steps is not None
            else int(ratio_steps * total_steps)
        )

        steps_accum = 0
        reward_accum = 0.0
        eff_episodes = 0.0  # counts full episodes as 1, partial as fraction

        for Ri, Li in zip(reversed(R), reversed(L)):
            room = window_steps - steps_accum
            if room <= 0:
                break
            if Li <= room:
                reward_accum += Ri
                steps_accum += Li
                eff_episodes += 1.0
            else:
                frac = room / Li
                reward_accum += Ri * frac
                steps_accum += room
                eff_episodes += frac
                break

        denom = eff_episodes if normalize == "per_episode" else steps_accum
        run_avgs.append(reward_accum / denom)

    return run_avgs


# def compute_run_avgs(metrics, ratio):
#     """
#     Compute average reward for each run over the last `ratio` fraction of episodes.
#     metrics: List[List[dict]] where inner list are episodes with keys incl. 'ep_return'
#     Returns: List[float] per-run averages
#     """
#     run_avgs = []
#     for run in metrics:
#         returns = [ep.get('ep_return', 0.0) for ep in run]
#         n = len(returns)
#         if n == 0:
#             run_avgs.append(0.0)
#             continue
#         idx = int(n * ratio)
#         idx = max(0, min(idx, n - 1))
#         run_avgs.append(float(np.mean(returns[idx:])))
#     return run_avgs


def find_trials(exp_dir):
    """Return a sorted list of trial directory names under exp_dir."""
    return sorted(d for d in os.listdir(exp_dir) if d.startswith(SYMLINK_PREFIX))


def check_incomplete_runs(exp_dir):
    """
    Identify trials missing metrics.pkl or agent.txt.
    Returns: (incomplete_list, num_complete)
    """
    num_complete = 0
    incomplete = []
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        files_ok = (
            os.path.isfile(os.path.join(trial_dir, METRICS_FILE)) and
            os.path.isfile(os.path.join(trial_dir, AGENT_FILE))
        )
        if not files_ok:
            incomplete.append(trial)
        else:
            num_complete += 1
    return incomplete, num_complete


def find_best_hyperparameters(exp_dir, ratio, auc_type):
    """
    Scan completed trials, compute run-level and overall average rewards,
    and return (overall_avg, run_avgs, agent_str, trial_name) for the best.
    """
    results = []  # (overall_avg, run_avgs, agent_str, trial)
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        metrics_path = os.path.join(trial_dir, METRICS_FILE)
        agent_path   = os.path.join(trial_dir, AGENT_FILE)
        if not (os.path.isfile(metrics_path) and os.path.isfile(agent_path)):
            continue
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        with open(agent_path, 'r') as f:
            agent_str = f.read().strip()
        if auc_type == "episode":
            run_avgs = compute_run_avgs_by_episodes_AUC(metrics, ratio=ratio)
        elif auc_type == "steps":
            run_avgs = compute_run_avgs_by_steps_AUC(metrics, ratio_steps=ratio)
        overall_avg = float(np.mean(run_avgs)) if run_avgs else 0.0
        results.append((overall_avg, run_avgs, agent_str, trial))

    if not results:
        return None
    return max(results, key=lambda x: x[0])


def load_args_yaml(trial_dir):
    """
    Load args.yaml (if present) from a trial directory and return the parsed dict.
    Returns None if missing/invalid.
    """
    args_path = os.path.join(trial_dir, ARGS_FILE)
    if not os.path.isfile(args_path):
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


def print_info_for_best_trial(exp_dir, ratio, auc_type, sort_keys=True, indent=2):
    """Pretty-print the best trial's info dict as JSON (for copy-paste)."""
    best = find_best_hyperparameters(exp_dir, ratio, auc_type)
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
    pretty = json.dumps(info_dict, indent=indent, sort_keys=sort_keys)
    print(pretty)
    


def collect_trial_records(exp_dir, ratio, auc_type):
    """
    Returns a list of per-trial records:
      {
        'trial': str,
        'overall_avg': float,
        'run_avgs': List[float],
        'params': Dict[str, Any]  # from args.yaml['info']
      }
    """
    records = []
    for trial in find_trials(exp_dir):
        trial_dir = os.path.join(exp_dir, trial)
        metrics_path = os.path.join(trial_dir, METRICS_FILE)
        agent_path   = os.path.join(trial_dir, AGENT_FILE)
        if not (os.path.isfile(metrics_path) and os.path.isfile(agent_path)):
            continue

        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        if auc_type == "episode":
            run_avgs = compute_run_avgs_by_episodes_AUC(metrics, ratio=ratio)
        elif auc_type == "steps":
            run_avgs = compute_run_avgs_by_steps_AUC(metrics, ratio_steps=ratio)
        
        overall_avg = float(np.mean(run_avgs)) if run_avgs else 0.0

        params = get_info_dict_from_trial(trial_dir) or {}
        records.append({
            'trial': trial,
            'overall_avg': overall_avg,
            'run_avgs': run_avgs,
            'params': params,
        })
    return records

def discover_hp_keys(exp_dir):
    """
    Read args.yaml for each trial and collect the union of keys under 'hp_search_space'.
    Supports either top-level 'hp_search_space' or nested under 'info'.
    """
    hp_keys = set()
    for trial in find_trials(exp_dir):
        td = os.path.join(exp_dir, trial)
        args = load_args_yaml(td)
        if not isinstance(args, dict):
            continue

        # Prefer top-level 'hp_search_space'
        hp = args.get("hp_search_space")
        if not isinstance(hp, dict):
            # Fallback: sometimes stored under info
            info = args.get("info", {})
            if isinstance(info, dict):
                hp = info.get("hp_search_space")

        if isinstance(hp, dict):
            hp_keys.update(hp.keys())

    return sorted(hp_keys)

# ----------------------------
# Sensitivity plotting (config-colored scatter)
# ----------------------------

def _config_signature(params: dict, target_key: str):
    """Signature for 'other hyper-params' (exclude the target key)."""
    return tuple(sorted((k, params[k]) for k in params.keys() if k != target_key))


def _human_label_from_sig(sig):
    if not sig:
        return "(others: none)"
    return ", ".join(f"{k}={v}" for k, v in sig)


def _order_x(values):
    """Order x values: numeric ascending first, then strings alphabetically."""
    def _cast(x):
        try:
            return float(x)
        except Exception:
            return None
    numeric = [( _cast(v), v) for v in values if _cast(v) is not None]
    nonnum  = [v for v in values if _cast(v) is None]
    numeric.sort(key=lambda t: t[0])
    return [v for _, v in numeric] + sorted(nonnum, key=lambda s: str(s).lower())


def plot_param_sensitivity_scatter_by_config(
    records,
    target_key: str,
    out_path: str,
    center_stat: str = "mean",    # or "median"
    legend_mode: str = "right",   # 'right' | 'below' | 'none'
    max_legend: int = 18,         # cap legend entries on-figure
    offset_width: float = 0.34,   # spread configs around each x
    jitter_width: float = 0.10,   # tiny jitter per run
):
    """
    For a given hyper-param (target_key):
      - x: distinct values of target_key
      - y: per-run rewards
      - color: unique 'other-params' configuration (all params except target_key)
      - small horizontal offsets avoid overlap of configs at same x
      - center line: mean/median of per-trial means at each x
      Legend is placed outside by default (right) to avoid covering dots.
    """
    # Bucket trials by x value, then by config signature
    by_x = defaultdict(lambda: defaultdict(list))  # x_val -> sig -> list(trial_records)
    all_sigs = set()
    x_values_raw = set()

    for r in records:
        if target_key not in r['params']:
            continue
        x_val = r['params'][target_key]
        sig = _config_signature(r['params'], target_key)
        by_x[x_val][sig].append(r)
        all_sigs.add(sig)
        x_values_raw.add(x_val)

    if not x_values_raw:
        print(f"[WARN] No trials carry param '{target_key}'. Skipping.")
        return

    # Order x values and prepare x positions
    x_values = _order_x(x_values_raw)
    spacing = 1.5  # increase this to spread them more (default = 1.0)
    x_pos = np.arange(len(x_values), dtype=float) * spacing

    # Stable color per config signature, global across x
    sig_list = sorted(list(all_sigs))
    cmap = plt.get_cmap("tab20")
    colors = {sig: cmap(i % 20) for i, sig in enumerate(sig_list)}

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(0)  # deterministic jitter

    center_vals = []  # center (mean/median) of trial means per x
    for i, xv in enumerate(x_values):
        cfg_dict = by_x[xv]
        cfg_sigs_here = sorted(cfg_dict.keys(), key=lambda s: str(s))
        k = len(cfg_sigs_here)
        if k == 0:
            center_vals.append(np.nan)
            continue

        # spread configs horizontally around x position
        offsets = [0.0] if k == 1 else np.linspace(-offset_width, offset_width, k)

        trial_means_all = []

        for off, sig in zip(offsets, cfg_sigs_here):
            trials = cfg_dict[sig]

            # trial means for this config at this x
            tmeans = [
                (float(np.mean(t['run_avgs'])) if t['run_avgs'] else t['overall_avg'])
                for t in trials
            ]
            trial_means_all.extend(tmeans)

            # per-run dots (stack run_avgs across trials at this x/config)
            ys = []
            for t in trials:
                ys.extend(t['run_avgs'])
            if not ys:
                continue

            jitter = (rng.random(len(ys)) - 0.5) * jitter_width
            ax.scatter(
                np.full(len(ys), x_pos[i] + off) + jitter,
                ys,
                s=28,
                alpha=0.78,
                color=colors[sig],
                edgecolor="none",
                label=_human_label_from_sig(sig),
            )

        # center across ALL configs/trials at this x (use trial means, not runs)
        if trial_means_all:
            val = float(np.median(trial_means_all)) if center_stat == "median" else float(np.mean(trial_means_all))
            center_vals.append(val)
        else:
            center_vals.append(np.nan)

    # Center line across x
    ax.plot(x_pos, center_vals, linewidth=2.5, color="black")

    # Legend: dedupe and truncate
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            uniq.append((h, lab))
            seen.add(lab)
    if len(uniq) > max_legend:
        uniq = uniq[:max_legend]
        overflow_note = f" (+{len(handles)-max_legend} more)"
    else:
        overflow_note = ""

    if legend_mode != "none" and uniq:
        ax.legend(
            *zip(*uniq),
            title="Other-params configs" + overflow_note,
            loc="center left" if legend_mode == "right" else "upper center",
            bbox_to_anchor=(1.02, 0.5) if legend_mode == "right" else (0.5, -0.12),
            ncol=1 if legend_mode == "right" else 2,
            fontsize=8,
            frameon=False,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_values], rotation=0)
    ax.set_xlabel(target_key)
    ax.set_ylabel("Reward (per run)")
    ttl_center = "median" if center_stat == "median" else "mean"
    ax.set_title(f"Sensitivity: {target_key}  (color=other params; line={ttl_center} of trial means)")
    ax.grid(True, linestyle="--", alpha=0.35)

    # os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    if legend_mode == "right":
        fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_all_param_sensitivities(exp_dir, ratio, out_dir, auc_type,
                                 center_stat="mean", legend_mode="right", max_legend=18):
    """
    Build records and generate config-colored scatter plots
    ONLY for keys listed in args.yaml['hp_search_space'] (top-level or under info).
    """
    records = collect_trial_records(exp_dir, ratio, auc_type)
    if not records:
        print("No completed trials to plot sensitivities.")
        return

    hp_keys = discover_hp_keys(exp_dir)
    if not hp_keys:
        print("No 'hp_search_space' found in any args.yaml — nothing to plot.")
        return

    # Guard against hp keys that aren't present in the per-trial info dicts
    present_keys = set()
    for r in records:
        present_keys.update(r["params"].keys())
    keys = [k for k in hp_keys if k in present_keys]

    if not keys:
        print("hp_search_space keys not present in trial info — nothing to plot.")
        return

    print(f"Plotting sensitivity only for hp_search_space keys: {keys}")
    os.makedirs(out_dir, exist_ok=True)
    for k in keys:
        out_path = os.path.join(out_dir, f"{k}_configs-{ratio}_ratio-{auc_type}_auc.png")
        plot_param_sensitivity_scatter_by_config(
            records,
            target_key=k,
            out_path=out_path,
            center_stat=center_stat,
            legend_mode=legend_mode,
            max_legend=max_legend,
        )


# ----------------------------
# CLI entry
# ----------------------------

def main(exp_dir, ratio, auc_type):
    # 1) Incomplete
    incomplete, num_complete = check_incomplete_runs(exp_dir)
    if incomplete:
        print("Incomplete trials (missing files):")
        for t in incomplete:
            print(f"  {t}")
    print(f"{num_complete} trials have metrics and agent info.")

    # 2) Best hyperparams
    best = find_best_hyperparameters(exp_dir, ratio, auc_type)
    if best is None:
        print("No completed trials found to analyze.")
        return
    overall_avg, run_avgs, agent_str, trial_name = best

    print("\nBest hyperparameter combination:")
    print(f"  Trial:          {trial_name}")
    print(f"  Agent details:  {agent_str}")
    print(f"  Overall avg:    {overall_avg:.6f}\n")

    print("Per-run average rewards:")
    for i, r in enumerate(run_avgs, 1):
        print(f"  Run {i}: {r:.6f}")

    # 3) Print INFO dict for copy-paste
    print_info_for_best_trial(exp_dir, ratio, auc_type)
    
    # 4) Dump the info and the trial name, run_avgs, overall_avg
    trial_dir = os.path.join(exp_dir, trial_name)
    best_info_dict = get_info_dict_from_trial(trial_dir)
    summary = {
        "_comment": "Summary of the best trial automatically generated for reference.",
        "trial_name": trial_name,
        "agent_details": agent_str,
        "overall_avg_reward": overall_avg,
        "run_average_rewards": run_avgs,
        "info": best_info_dict,
    }

    summary_path = os.path.join(exp_dir, f"best_trial_summary-{ratio}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\nBest trial summary saved to: {summary_path}")


if __name__ == '__main__':
    # --- Configuration ---
    exp_dir = "Runs/Sweep/BigCurriculumEnv-v0_/DQN/_seed[1]"
    ratio   = 0.9 # average the last ratio --> 0.0: only last  ---  1.0: all
    auc_type = "steps" # steps or episode
    # ---------------------

    # Optional summary/diagnostics:
    main(exp_dir, ratio, auc_type)

    # Plot sensitivities for all keys found in info:
    # plot_all_param_sensitivities(
    #     exp_dir, ratio, out_dir=exp_dir, auc_type=auc_type,
    #     center_stat="mean",
    #     legend_mode="right",   # 'right' puts legend outside; try 'below' or 'none'
    #     max_legend=100
    # )