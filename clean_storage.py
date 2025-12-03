#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

def find_and_delete(root: Path, condition_name: str, target_name: str,
                    dry_run: bool = True, ignore_case: bool = False):
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Normalize names if case-insensitive
    def norm(s: str) -> str:
        return s.lower() if ignore_case else s

    n_condition = norm(condition_name)
    n_target = norm(target_name)

    deleted = 0
    candidates = 0
    errors = 0

    # rglob("*") with .iterdir recursion via os.walk-like pattern
    for dirpath in [p for p in root.rglob("*") if p.is_dir()] + [root]:
        try:
            # Build a quick name -> Path map for files in this dir
            entries = list(dirpath.iterdir())
            file_by_name = {norm(p.name): p for p in entries if p.is_file()}

            if n_condition in file_by_name:
                # We have the condition file in this folder
                candidates += 1
                if n_target in file_by_name:
                    target_path = file_by_name[n_target]
                    print(f"[MATCH] {dirpath} has '{condition_name}'. "
                          f"{'Would delete' if dry_run else 'Deleting'}: {target_path}")
                    if not dry_run:
                        try:
                            target_path.unlink()
                            deleted += 1
                        except Exception as e:
                            errors += 1
                            print(f"  -> ERROR deleting {target_path}: {e}", file=sys.stderr)
                else:
                    # Condition met but no target file to delete
                    print(f"[MATCH] {dirpath} has '{condition_name}', but '{target_name}' not found.")
        except PermissionError as e:
            errors += 1
            print(f"[SKIP] Permission denied: {dirpath} ({e})", file=sys.stderr)
        except Exception as e:
            errors += 1
            print(f"[SKIP] Error reading {dirpath}: {e}", file=sys.stderr)

    print("\nSummary:")
    print(f"  Folders with condition file: {candidates}")
    print(f"  Files deleted: {deleted} {'(dry-run: 0 actually deleted)' if dry_run else ''}")
    print(f"  Errors: {errors}")

def main():
    parser = argparse.ArgumentParser(
        description="If a directory contains CONDITION file, delete TARGET file in that same directory."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan")
    parser.add_argument("--condition", required=True,
                        help="File name that must exist in a directory to trigger deletion")
    parser.add_argument("--target", required=True,
                        help="File name to delete when condition is met")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete files (default is dry-run)")
    parser.add_argument("--ignore-case", action="store_true",
                        help="Match file names case-insensitively")
    args = parser.parse_args()

    find_and_delete(
        root=args.root,
        condition_name=args.condition,
        target_name=args.target,
        dry_run=(not args.apply),
        ignore_case=args.ignore_case,
    )

if __name__ == "__main__":
    main()