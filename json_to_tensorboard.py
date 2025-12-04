import os
import json
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def has_tensorboard_files(dirpath: str) -> bool:
    """Return True if any TensorBoard event file exists in dirpath."""
    for fname in os.listdir(dirpath):
        if ".tfevents." in fname:
            return True
    return False


def process_exp_dir_streaming(exp_dir: str):
    """
    Given a single experiment directory that contains metrics.jsonl
    and no TensorBoard events yet, stream metrics.jsonl -> TensorBoard
    in a single pass (no accumulation).
    """
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")

    print(f"\n[Exp] {exp_dir}")
    print(f"  Reading metrics from: {metrics_path}")
    print(f"  Writing TensorBoard logs to: {exp_dir}")

    if not os.path.isfile(metrics_path):
        print("  metrics.jsonl not found, skipping.")
        return

    file_size = os.path.getsize(metrics_path)
    if file_size == 0:
        print("  metrics.jsonl is empty, skipping.")
        return

    # Slightly larger buffers to avoid flushing constantly.
    writer = SummaryWriter(log_dir=exp_dir, flush_secs=30, max_queue=10_000)

    # Single pass: read → write
    with open(metrics_path, "r") as f, tqdm(
        total=file_size,
        desc="  Converting",
        unit="B",
        unit_scale=True,
        leave=False,
    ) as pbar:
        for line in f:
            pbar.update(len(line))
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            step = rec["step"]
            tag = rec["tag"]
            metrics = rec["metrics"]

            # Stream directly to TB
            for key, value in metrics.items():
                name = f"{tag}/{key}"
                writer.add_scalar(name, value, step)

    writer.close()
    print("  Conversion complete ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="Runs/",
        help="Top-level directory to recursively search for metrics.jsonl",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    print(f"Root directory: {root_dir}")

    # 1) Discover all experiment directories needing conversion
    exp_dirs_to_process = []

    print("\nScanning for metrics.jsonl without TensorBoard logs...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "metrics.jsonl" in filenames and not has_tensorboard_files(dirpath):
            exp_dirs_to_process.append(dirpath)

    if not exp_dirs_to_process:
        print("No directories found with metrics.jsonl lacking TensorBoard logs. Nothing to do.")
        return

    print(f"Found {len(exp_dirs_to_process)} experiment(s) to process.")

    # 2) Process each experiment directory
    for exp_dir in tqdm(exp_dirs_to_process, desc="Processing experiments"):
        try:
            process_exp_dir_streaming(exp_dir)
        except Exception as e:
            print(f"  [Warning] Failed to process {exp_dir}: {e}")

    print("\nAll done ✓")


if __name__ == "__main__":
    main()