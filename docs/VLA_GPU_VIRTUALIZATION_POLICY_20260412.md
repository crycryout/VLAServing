# VLA Workload-Aware GPU Virtualization Policy

This note documents a new workload-aware GPU resource partitioning policy implemented in:

- [bench_vla_gpu_virtualization_policy.py](../src/bench_vla_gpu_virtualization_policy.py)

Results are written to:

- [vla_gpu_virtualization_policy_20260412.json](../results/vla_gpu_virtualization_policy_20260412.json)
- [pi05_vla_serving_autoh25_50_phase_shift_20260413.json](../results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json)

## Goal

Make the GPU scheduler explicitly aware of VLA workload structure:

- requests are predictable rather than fully reactive
- different robots bind to different fine-tuned models
- model-state residency and prefetch matter as much as compute time
- action exhaustion is the hard constraint
- AutoHorizon is a dynamic soft target

## Policy

The policy keeps:

- one dedicated shell for the `30Hz` model
- one dedicated shell for the `20Hz` model
- one shared shell for the two `10Hz` models

It also supports:

- frequency-aware partial residency for the shared models
- predictive prefetch for missing model state
- action-consumption-aware phase shifting

## Phase Modes

Two modes are evaluated.

`strict_horizon`

- each chunk must be served exactly at the sampled AutoHorizon
- no extra flexibility is used

`phase_shift`

- the system may serve a chunk anywhere in a safe action-consumption window
- the lower bound of that window is:
  - `25` for `Pi0.5` under the current `{25, 50}` control semantics
  - `min(horizon, 6)` for `GR00T N1.6`

This is the workload-aware extension:

- for long-horizon chunks, the scheduler may shift the next request phase to align with shell availability and prefetch completion

## Pi0.5 25/50 Semantic Variant

We also reran the Pi0.5 serving policy under a stricter control-side interpretation:

- any sampled AutoHorizon below `25` is treated as `25`
- any sampled AutoHorizon above `25` is treated as `50`
- the legal early-replan window is therefore `[25, 50]`

This variant is implemented in:

- [bench_pi05_vla_serving_autoh25_50_phase_shift.py](../src/bench_pi05_vla_serving_autoh25_50_phase_shift.py)

and writes:

- [pi05_vla_serving_autoh25_50_phase_shift_20260413.json](../results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json)

## Key Metric Definitions

Two latency metrics are reported.

`request_to_result_ms`

- from actual request issue time to returned result
- under this policy, request issue happens when the scheduler intentionally triggers the replan

`chunk_elapsed_ms`

- from chunk start to completion of the next inference
- this includes control-side waiting caused by larger chosen horizons

For VLA real-time claims, `request_to_result_ms` is the relevant metric.

## Main Findings

### Pi0.5

Under the current `{25, 50}` semantic variant:

- fixed-4 still stays at:
  - zero hard misses
  - zero action exhaustion
  - `request-to-result p95 ≈ 43.21ms`
  - perfect fixed-4 fleet/min scores
- admission improves materially:
  - `mean_admitted_total: 22.33 -> 32.67`
  - `mean_fleet_score: 0.9862 -> 0.9934`
  - `mean_min_robot_score: 0.9483 -> 0.9644`
  - `mean_miss_autohorizon_ratio: 0.0947 -> 0.0420`

Interpretation:

- for Pi0.5, the main gain here is not aggressive phase movement itself
- the real gain comes from changing the control semantics so the legal request window becomes `[25, 50]`
- once that wider window exists, the same predictive-residency and prefetch runtime can admit substantially more robots without hurting request latency
- even under this new setting, `phase_shift` still does not outperform `strict_horizon` on admission count; the gain comes from the wider legal window, not from extra phase motion

### GR00T N1.6

Under the current `30/20/10/10` setup:

- `strict_horizon` is better than `phase_shift`
- `phase_shift` reduces fleet/min score and increases AutoHorizon miss ratio
- `request_to_result p95` stays near raw inference time, but the control-side chunk timing gets worse

Interpretation:

- `GR00T N1.6` has shorter chunk budgets and tighter horizon structure
- extra phase flexibility is not automatically useful
- for this workload, preserving stricter horizon timing works better than aggressive phase movement

## Conclusion

The policy is implemented and validated, but the experiments show a workload-dependent result:

- `Pi0.5`: predictive residency + prefetch already captures most of the benefit; phase shifting adds little in the current fixed four-robot setting
- `Pi0.5 (25/50 semantic variant)`: widening the legal replan window to `[25, 50]` significantly improves admission capacity while keeping `request-to-result p95` unchanged
- `GR00T N1.6`: phase shifting can hurt because the horizon slack is much tighter

So the main system contribution that survives both workloads is:

- predictive model-state residency
- predictive prefetch
- shell-aware resource partitioning
- control-semantic window design

while phase shifting should remain an optional optimization rather than a universal default.
