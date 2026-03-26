"""
analyze_classifiers.py
======================
Average training curves and confusion matrices across multiple classifier runs.

Directory structure expected:
  <base_dir>/<option_run>/<exp_tag>/training_history.json

  base_dir  : agent-level classifier dir, e.g.
              Runs/Classifier/MiniGrid-.../FullyObs_.../VQOptionCritic
  option_run: one folder per data-collection run, filtered by --include/--exclude
              e.g. Options_LimitedColor_emb[uniform-d8]_43_seed[43]
  exp_tag   : the specific classifier config to analyse, e.g.
              Feat[delta_last]_KL[0.0]_ReprDim[8]

Usage examples:
  python analyze_classifiers.py \\
      --base_dir Runs/Classifier/MiniGrid-UnlockPickupLimitedColor-v0_/FullyObs_OneHotImageDirCarry/VQOptionCritic \\
      --include uniform-d8 \\
      --exp_tag "Feat[delta_last]_KL[0.0]_ReprDim[8]" \\
      --name_tag uniform_d8_KL0

  # Multiple include/exclude conditions:
  python analyze_classifiers.py \\
      --base_dir Runs/Classifier/.../VQOptionCritic \\
      --include uniform-d8 --exclude seed[34] \\
      --exp_tag "Feat[delta_last]_KL[0.01]_ReprDim[64]" \\
      --name_tag uniform_d8_KL0.01_dim64

"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    for s in include:
        if s not in name:
            return False
    for s in exclude:
        if s in name:
            return False
    return True


def collect_runs(base_dir: str, include: list[str], exclude: list[str], exp_tag: str) -> list[Path]:
    """
    Returns list of run_dirs that match:
      base_dir/<option_run>/<exp_tag>/
    where <option_run> passes the include/exclude filter and
    <exp_tag>/training_history.json exists.
    """
    base = Path(base_dir)
    run_dirs = []

    for option_run_dir in sorted(base.iterdir()):
        if not option_run_dir.is_dir():
            continue
        if not _matches(option_run_dir.name, include, exclude):
            continue
        exp_dir = option_run_dir / exp_tag
        if exp_dir.is_dir() and (exp_dir / "training_history.json").exists():
            run_dirs.append(exp_dir)
        else:
            if exp_dir.is_dir():
                print(f"  [warn] {option_run_dir.name}/{exp_tag}: no training_history.json")
            # silently skip option_run dirs that don't have the exp_tag at all

    return run_dirs


# ---------------------------------------------------------------------------
# training curves
# ---------------------------------------------------------------------------

def load_history(run_dir: Path) -> list[dict]:
    with open(run_dir / "training_history.json") as f:
        return json.load(f)


def average_histories(histories: list[list[dict]]) -> dict:
    """
    Align by epoch index and compute mean ± std per metric.
    Returns dict: metric -> (mean_array, std_array, epochs_array)
    """
    min_len = min(len(h) for h in histories)
    metrics = [k for k in histories[0][0].keys() if k != "epoch"]
    epochs = np.array([histories[0][i]["epoch"] for i in range(min_len)])

    result = {}
    for m in metrics:
        vals = np.array([[h[i][m] for i in range(min_len)] for h in histories])
        result[m] = (vals.mean(axis=0), vals.std(axis=0), epochs)
    return result


def plot_average_training_curves(run_dirs: list[Path], exp_tag: str, name_tag: str, save_dir: str = "Runs/Figures"):
    os.makedirs(save_dir, exist_ok=True)

    histories = []
    for rd in run_dirs:
        try:
            histories.append(load_history(rd))
        except Exception as e:
            print(f"  [warn] skipping {rd}: {e}")

    if not histories:
        print("  No valid training histories found.")
        return

    stats = average_histories(histories)
    print(f"  Averaged {len(histories)} training histories.")

    metrics = list(stats.keys())

    # pair train/val metrics into rows
    metric_pairs = []
    used = set()
    for m in metrics:
        if m in used:
            continue
        counterpart = (
            m.replace("train_", "val_") if m.startswith("train_")
            else m.replace("val_", "train_") if m.startswith("val_")
            else None
        )
        if counterpart and counterpart in metrics and counterpart not in used:
            metric_pairs.append((m, counterpart))
            used.add(m)
            used.add(counterpart)
        else:
            metric_pairs.append((m,))
            used.add(m)

    n_rows = len(metric_pairs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)
    fig.suptitle(f"{exp_tag}\n({len(histories)} runs)", fontsize=10)

    for row, pair in enumerate(metric_pairs):
        ax = axes[row][0]
        for metric in pair:
            if metric not in stats:
                continue
            mean, std, epochs = stats[metric]
            ls = "--" if metric.startswith("val_") else "-"
            label = metric
            ax.plot(epochs, mean, linestyle=ls, label=label)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)

        title_metric = pair[0].replace("train_", "").replace("val_", "")
        ax.set_title(title_metric)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{name_tag}_training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# confusion matrices — both train and val as subplots
# ---------------------------------------------------------------------------

def _load_cm(run_dir: Path, split: str) -> np.ndarray | None:
    """Try best then final checkpoint for the given split."""
    for tag in ("best", "final"):
        npy = run_dir / f"confusion_matrix_{tag}_{split}.npy"
        if npy.exists():
            return np.load(npy)
    return None


def _normalize_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    return cm.astype(float) / row_sums


def plot_confusion_matrices(run_dirs: list[Path], exp_tag: str, name_tag: str, save_dir: str = "Runs/Figures"):
    """
    Two-row subplot: train (top) and val (bottom).
    Each row has mean (left) and std (right) of row-normalised CMs.
    """
    os.makedirs(save_dir, exist_ok=True)

    splits = ["train", "val"]
    cms_by_split = {}
    for split in splits:
        cms = []
        for rd in run_dirs:
            cm = _load_cm(rd, split)
            if cm is not None:
                cms.append(cm)
        cms_by_split[split] = cms

    # check we have at least one split
    if not any(cms_by_split[s] for s in splits):
        print("  [skip] no .npy confusion matrices found (re-run training to generate them)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{exp_tag}", fontsize=11)

    for row, split in enumerate(splits):
        cms = cms_by_split[split]
        if not cms:
            for col in range(2):
                axes[row][col].set_visible(False)
            continue

        normed = [_normalize_cm(cm) for cm in cms]
        mean_cm = np.mean(normed, axis=0)
        std_cm  = np.std(normed,  axis=0)
        n = mean_cm.shape[0]
        ticks = np.arange(n)

        for col, (mat, subtitle) in enumerate([(mean_cm, "Mean"), (std_cm, "Std")]):
            ax = axes[row][col]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{split}  —  {subtitle}  ({len(cms)} runs)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticklabels(ticks, fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{name_tag}_confusion_matrices.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Agent-level Classifier dir, e.g. Runs/Classifier/.../VQOptionCritic")
    parser.add_argument("--exp_tag", type=str, required=True,
                        help="Classifier config subfolder, e.g. 'Feat[delta_last]_KL[0.0]_ReprDim[8]'")
    parser.add_argument("--name_tag", type=str, required=True,
                        help="Stem for saved figure filenames")
    parser.add_argument("--include", type=str, nargs="*", default=[],
                        help="Substrings that must appear in option_run directory name")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="Substrings that must NOT appear in option_run directory name")
    return parser.parse_args()


def main():
    args = parse()

    print(f"Base dir : {args.base_dir}")
    print(f"Exp tag  : {args.exp_tag}")
    print(f"Name tag : {args.name_tag}")
    print(f"Include  : {args.include}")
    print(f"Exclude  : {args.exclude}")
    print(f"Save dir : Runs/Figures")
    print()

    run_dirs = collect_runs(args.base_dir, args.include, args.exclude, args.exp_tag)
    if not run_dirs:
        print("No runs found. Check --base_dir, --exp_tag, and --include/--exclude.")
        return

    print(f"Found {len(run_dirs)} run(s):")
    for rd in run_dirs:
        print(f"  {rd}")
    print()

    print("Plotting average training curves ...")
    plot_average_training_curves(run_dirs, args.exp_tag, args.name_tag)
    print()

    print("Averaging confusion matrices (train + val) ...")
    plot_confusion_matrices(run_dirs, args.exp_tag, args.name_tag)


if __name__ == "__main__":
    main()
