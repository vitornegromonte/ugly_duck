#!/usr/bin/env python3
"""
LoRC — Hyperparameter sweep.

Generates the Cartesian grid from `sweep_config.yaml`, applies skip rules,
then runs each combination through `lore.run_pipeline`. Supports resumption
(skips completed runs) and parallel execution via `--parallel N`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_grid(sweep: dict) -> list[dict]:
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def matches_skip(combo: dict, skip_rule: dict) -> bool:
    for k, v in skip_rule.items():
        combo_val = combo.get(k)
        if isinstance(v, list):
            if combo_val not in v:
                return False
        elif combo_val != v:
            return False
    return True


def filter_skipped(combos: list[dict], skip_rules: list[dict]) -> list[dict]:
    return [
        c for c in combos
        if not any(matches_skip(c, r) for r in skip_rules)
    ]


def combo_id(combo: dict) -> str:
    parts = []
    for k in sorted(combo.keys()):
        v = combo[k]
        if isinstance(v, bool):
            v = str(v).lower()
        parts.append(f"{k}={v}")
    return "_".join(parts)


def is_completed(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "lore_results.json"))


def _run_single(args: tuple) -> dict:
    combo, env, force = args
    run_id = combo_id(combo)
    base_dir = env["output_dir"]
    run_dir = os.path.join(base_dir, run_id)

    if not force and is_completed(run_dir):
        return {"id": run_id, "status": "skipped"}

    os.makedirs(run_dir, exist_ok=True)

    cmd = [sys.executable, "-m", "lore.run_pipeline"]
    for k, v in combo.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k.replace('_', '-')}")
        else:
            cmd.append(f"--{k.replace('_', '-')}")
            cmd.append(str(v))
    cmd.extend([
        "--model", env["model"],
        "--device", env["device"],
        "--seed", str(env["seed"]),
        "--output-dir", run_dir,
    ])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        elapsed = time.time() - t0
        t_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        if result.returncode != 0:
            print(f"  FAIL {run_id} ({t_str})")
            if result.stderr:
                print(f"       {result.stderr[-300:]}")
            return {"id": run_id, "status": "failed", "elapsed": elapsed, "stderr": result.stderr[-300:]}
        else:
            print(f"  DONE {run_id} ({t_str})")
            return {"id": run_id, "status": "completed", "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        print(f"  FAIL {run_id} (timeout)")
        return {"id": run_id, "status": "timeout"}
    except Exception as e:
        print(f"  FAIL {run_id}: {e}")
        return {"id": run_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="LoRC hyperparameter sweep")
    parser.add_argument("--config", default="lore/sweep_config.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments")
    parser.add_argument("--parallel", type=int, default=1, help="Max parallel processes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sweep_params = cfg["sweep"]
    skip_rules = cfg.get("skip", [])
    env = cfg["env"]

    all_combos = expand_grid(sweep_params)
    filtered = filter_skipped(all_combos, skip_rules)

    print("=" * 60)
    print("LoRC — Hyperparameter Sweep")
    print(f"  Total combinations:  {len(all_combos)}")
    print(f"  After skip rules:   {len(filtered)}")
    print(f"  Parallel processes: {args.parallel}")
    print(f"  Output dir:         {env['output_dir']}")
    print("=" * 60)

    if args.dry_run:
        print("Combinations:")
        for i, combo in enumerate(filtered):
            print(f"  {i+1:>4}. {combo_id(combo)}")
        print(f"Total: {len(filtered)}")
        return

    task_args = [(combo, env, args.force) for combo in filtered]

    if args.parallel <= 1:
        results = []
        for i, ta in enumerate(task_args):
            print(f"\n--- Run {i+1}/{len(task_args)} ---")
            results.append(_run_single(ta))
    else:
        with multiprocessing.Pool(args.parallel) as pool:
            results = pool.map(_run_single, task_args)

    n_completed = sum(1 for r in results if r["status"] == "completed")
    n_failed = sum(1 for r in results if r["status"] != "completed" and r["status"] != "skipped")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n{'=' * 60}")
    print(f"  Completed: {n_completed}  Failed: {n_failed}  Skipped: {n_skipped}")
    print(f"{'=' * 60}")

    summary_path = os.path.join(env["output_dir"], "sweep_summary.json")
    rows = []
    for combo in filtered:
        run_dir = os.path.join(env["output_dir"], combo_id(combo))
        results_file = os.path.join(run_dir, "lore_results.json")
        if os.path.isfile(results_file):
            with open(results_file) as f:
                data = json.load(f)
            rows.append({"id": combo_id(combo), "params": combo, "results": data})
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Summary table: {summary_path}")


if __name__ == "__main__":
    main()
