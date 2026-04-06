# VLAServing

This repo collects the Pi0.5 / PI05 and GR00T N1.6 serving experiments developed in the local workspace.

## Main contents

- `src/lerobot/policies/pi05/`
  - PI0.5 / PI05 policy code and AutoHorizon selector port.
- `src/lerobot/eval/simulate_pi05_global_reservation_serving.py`
  - Main retained PI0.5 reservation scheduler.
- `src/lerobot/eval/run_pi05_global_reservation_trials.py`
  - Random-arrival runner for the retained PI0.5 baseline.
- `src/lerobot/eval/bench_pi05_four_model_residency_prefetch_system.py`
  - Four-model PI0.5 GPU-shell residency + predictive prefetch simulator.
- `src/lerobot/eval/bench_pi05_residency_prefetch_admission_bound.py`
  - Admission-control simulation with fixed model-frequency binding.
- `src/lerobot/eval/bench_pi05_autohorizon_reservation_prefetch.py`
  - Latest PI0.5 GPU-virtualization simulator with:
    - AutoHorizon
    - reservation-based compute slot assignment
    - model residency / prefetch modeling
    - fixed model-frequency binding
- `src/gr00t/eval/`
  - GR00T N1.6 serving simulation code.

## PI0.5 retained baseline

Retained baseline:

- single-stage compiled full E2E inference
- no queueing
- no deferred switch after new actions are ready
- hard rule: no robot may exhaust `50` actions before the next chunk is ready

Baseline result:

- `results/pi05_global_reservation_final_cfg055_fast10_alignment_20260329.json`
- total admitted robots: `120`
- mean admitted per group: `12.0`
- `reply_over_chunk_actions_count = 0`
- `miss_autohorizon_ratio = 11.52%`

## PI0.5 GPU virtualization line

The latest PI0.5 system idea in this repo is:

- treat PI0.5 VLA serving as a GPU-virtualization problem
- bind each robot frequency to a fine-tuned model
- reserve GPU compute slots using AutoHorizon targets
- model high-frequency robots with more residency
- use predictive prefetch for lower-frequency models

Main code:

- `src/lerobot/eval/bench_pi05_four_model_residency_prefetch_system.py`
- `src/lerobot/eval/bench_pi05_residency_prefetch_admission_bound.py`
- `src/lerobot/eval/bench_pi05_autohorizon_reservation_prefetch.py`

Main results:

- `results/pi05_four_model_residency_prefetch_system_20260406.json`
  - fixed `30Hz + 20Hz + 10Hz + 10Hz`
  - `30Hz/20Hz` resident shells
  - two `10Hz` models share one shell with predictive prefetch
  - `hard_deadline_miss_count = 0`
- `results/pi05_residency_prefetch_admission_bound_20260406.json`
  - fixed model-frequency binding
  - average admitted total: `9.0` over 3 quick groups
  - `hard_miss_count = 0`
- `results/pi05_autohorizon_reservation_prefetch_20260406.json`
  - latest AutoHorizon-aware reservation / residency / prefetch version
  - fixed four-robot case:
    - `hard_miss_count = 0`
    - `reply_over_chunk_actions_count = 0`
    - `miss_autohorizon_count = 0`
    - `fleet_score = 1.0`
    - `min_robot_score = 1.0`

Detailed Chinese explanation:

- `docs/PI05_GPU_VIRTUALIZATION.md`

## GR00T N1.6

This repo also contains GR00T N1.6 serving experiments derived from the same reservation-based idea.

Representative result:

- `results/groot_n1d6_global_reservation_final_cfg040_bins16_fast_20260329.json`

## Notes

- This repo is a curated experiment repo, not a full standalone LeRobot distribution.
- Many scripts assume the surrounding local environment used in the experiments.
- Some result JSON files are retained as experiment artifacts for reproducibility.
