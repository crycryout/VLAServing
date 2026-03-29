# VLAServing

Pi05-related code snapshot extracted from the local LeRobot workspace and pushed here as a standalone reference.

## Included code

- `src/lerobot/policies/pi05/`
  - LeRobot PI0.5 / PI05 policy files.
  - Includes the `official_bidir` AutoHorizon selector port in `autohorizon.py`.
- `src/lerobot/eval/libero_native_eval_pi05.py`
  - Native LIBERO evaluation entrypoint for PI05.
- `src/lerobot/eval/run_libero_native_autohorizon_suites.py`
  - Batch runner for LeRobot PI05 AutoHorizon suite evaluations.
- `src/lerobot/eval/fit_pi05_autohorizon_simulator.py`
  - Fits an empirical horizon simulator from recorded PI05 AutoHorizon event logs.
- `src/lerobot/eval/bench_pi05_component_latency.py`
  - Component latency benchmark used for PI05 step/compile timing checks.
- `src/lerobot/eval/simulate_pi05_global_reservation_serving.py`
  - Current mainline serving simulator for PI05.
  - No queueing.
  - New actions switch immediately when ready.
  - `50`-action exhaustion is treated as a hard failure.
- `src/lerobot/eval/run_pi05_global_reservation_trials.py`
  - Random-arrival experiment runner for the mainline global-reservation scheduler.

## Mainline serving version

The current retained PI05 serving mainline in this repo is:

- `single_stage_compiled_full_e2e_global_reservation`
- `denoising_step = 1`
- `torch.compile = true`
- no queueing
- no deferred action-cache switch
- hard rule: a robot must never exhaust `50` actions before the next chunk is ready

This is the version kept after restoring the best PI05 result path from local experiments.

## Included result artifacts

- `results/pi05_autohorizon_simulator_fit_20260329.json`
  - Empirical PI05 AutoHorizon simulator fit.
- `results/lerobot_p50_step1_full_e2e_batch_sweep_compile_dynamic_20260327_summary.json`
  - `torch.compile` batch latency summary for PI05 step-1 full E2E inference.
- `results/pi05_global_reservation_final_cfg055_fast10_alignment_20260329.json`
  - Main 10-group PI05 serving experiment output.
- `results/pi05_global_reservation_horizon_alignment_cfg055_fast10_20260329.json`
  - Per-horizon early / exact / late alignment summary.

## Key mainline result

From `results/pi05_global_reservation_final_cfg055_fast10_alignment_20260329.json`:

- total admitted robots: `120`
- mean admitted per group: `12.0`
- admitted frequencies:
  - `5Hz: 29`
  - `10Hz: 17`
  - `15Hz: 19`
  - `20Hz: 18`
  - `25Hz: 24`
  - `30Hz: 13`
- `reply_over_chunk_actions_count = 0`
- `miss_autohorizon_ratio = 11.52%`

## Notes

- This repo is a curated code snapshot, not a full standalone LeRobot distribution.
- Some scripts still assume the original LeRobot package layout around them.
- Experimental PI05 schedulers that were later rejected are intentionally not included as mainline code here.
