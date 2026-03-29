#!/usr/bin/env python3
"""Fit an empirical AutoHorizon simulator from LeRobot PI0.5 event logs."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

import tyro


@dataclass
class Config:
    run_root: str = "/root/_runs/lerobot_p50_autohorizon_suites_official_bidir_20260327_223304"
    output_path: str = "/root/autodl-tmp/pi05_autohorizon_simulator_fit_20260329.json"


def main() -> None:
    cfg = tyro.cli(Config)
    root = Path(cfg.run_root)
    files = sorted(root.glob("*/autohorizon_events.csv"))
    start = Counter()
    trans = defaultdict(Counter)
    counts = Counter()
    total = 0

    for path in files:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: (r["task_group"], int(r["task_id"]), int(r["episode_index"]), int(r["decision_index"])))
        prev_key = None
        prev_h = None
        for row in rows:
            key = (row["task_group"], int(row["task_id"]), int(row["episode_index"]))
            h = int(row["execution_horizon"])
            counts[h] += 1
            total += 1
            if key != prev_key:
                start[h] += 1
            else:
                trans[int(prev_h)][h] += 1
            prev_key = key
            prev_h = h

    output = {
        "run_root": str(root),
        "total_events": total,
        "states": sorted(counts.keys()),
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "frequencies": {str(k): float(v / total) for k, v in sorted(counts.items())},
        "mean_horizon": float(sum(k * v for k, v in counts.items()) / total) if total else 0.0,
        "start_counts": {str(k): int(v) for k, v in sorted(start.items())},
        "start_probs": {
            str(k): float(v / sum(start.values())) for k, v in sorted(start.items())
        } if start else {},
        "transition_counts": {
            str(k): {str(k2): int(v2) for k2, v2 in sorted(v.items())} for k, v in sorted(trans.items())
        },
        "transition_probs": {
            str(k): {str(k2): float(v2 / sum(v.values())) for k2, v2 in sorted(v.items())} for k, v in sorted(trans.items())
        },
    }

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
