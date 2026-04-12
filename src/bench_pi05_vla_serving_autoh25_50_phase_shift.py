#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import bench_vla_gpu_virtualization_policy as base


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "pi05_vla_serving_autoh25_50_phase_shift_20260413.json"


def collapse_pi05_horizon_to_25_50(horizon: dict[str, object]) -> dict[str, object]:
    def map_state(h: int) -> int:
        return 25 if h < 25 else 50

    start_mass = {25: 0.0, 50: 0.0}
    for state, prob in zip(horizon["start_states"], horizon["start_probs"]):
        start_mass[map_state(int(state))] += float(prob)

    transition_mass: dict[int, dict[int, float]] = {25: {25: 0.0, 50: 0.0}, 50: {25: 0.0, 50: 0.0}}
    transition = horizon["transition"]
    for src, (states, probs) in transition.items():
        mapped_src = map_state(int(src))
        for state, prob in zip(states, probs):
            transition_mass[mapped_src][map_state(int(state))] += float(prob)

    out_transition = {}
    for src in (25, 50):
        states = []
        probs = []
        total = sum(transition_mass[src].values())
        for dst in (25, 50):
            mass = transition_mass[src][dst]
            if mass <= 0.0:
                continue
            states.append(dst)
            probs.append(mass / total if total > 0.0 else 0.0)
        out_transition[src] = (
            base.np.asarray(states, dtype=base.np.int64),
            base.np.asarray(probs, dtype=base.np.float64),
        )

    mean_horizon = 25.0 * start_mass[25] + 50.0 * start_mass[50]
    return {
        "start_states": base.np.asarray([25, 50], dtype=base.np.int64),
        "start_probs": base.np.asarray([start_mass[25], start_mass[50]], dtype=base.np.float64),
        "transition": out_transition,
        "mean_horizon": float(mean_horizon),
    }


def main() -> None:
    original_horizon = base.WORKLOADS["pi05"]["horizon"]
    original_floor = base.WORKLOADS["pi05"]["phase_shift_floor_actions"]
    try:
        base.WORKLOADS["pi05"]["horizon"] = collapse_pi05_horizon_to_25_50(original_horizon)
        base.WORKLOADS["pi05"]["phase_shift_floor_actions"] = 25
        result = {
            "meta": {
                "source_script": str(Path(base.__file__).resolve()),
                "result_file": str(OUT),
                "notes": [
                    "Pi05 AutoHorizon states are collapsed to {25, 50}.",
                    "Any original horizon below 25 is treated as 25; any original horizon above 25 is treated as 50.",
                    "Under phase_shift, request service may trigger after any consumed action in the closed window [25, 50].",
                    "This reruns the workload-aware VLA serving policy only for Pi05.",
                ],
                "seeds": list(base.SEEDS),
                "predict_duration_s": base.PREDICT_DURATION_S,
                "truth_duration_s": base.TRUTH_DURATION_S,
            },
            "pi05_autoh25_50": base.evaluate_workload("pi05"),
        }
        OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps({"out": str(OUT)}, indent=2))
    finally:
        base.WORKLOADS["pi05"]["horizon"] = original_horizon
        base.WORKLOADS["pi05"]["phase_shift_floor_actions"] = original_floor


if __name__ == "__main__":
    main()
