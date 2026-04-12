# VLA Workload-Aware GPU Virtualization Policy

This note documents a new workload-aware GPU resource partitioning policy implemented in:

- [bench_vla_gpu_virtualization_policy.py](../src/bench_vla_gpu_virtualization_policy.py)

Results are written to:

- [vla_gpu_virtualization_policy_20260412.json](../results/vla_gpu_virtualization_policy_20260412.json)

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
  - `min(horizon, 20)` for `Pi0.5`
  - `min(horizon, 6)` for `GR00T N1.6`

This is the workload-aware extension:

- for long-horizon chunks, the scheduler may shift the next request phase to align with shell availability and prefetch completion

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

Under the current `30/20/10/10` four-robot setup:

- `strict_horizon` and `phase_shift` both achieve:
  - zero hard misses
  - zero action exhaustion
  - `request_to_result p95 ≈ 43.21ms`
  - perfect fixed-4 fleet/min scores

Admission result:

- both modes admit about `22.33` robots on average under the configured quality thresholds

Interpretation:

- with the current three-shell predictive-prefetch design, the `Pi0.5` four-robot case is already not bottlenecked by phase alignment
- the additional phase-shift flexibility does not improve the fixed four-robot setup

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
- `GR00T N1.6`: phase shifting can hurt because the horizon slack is much tighter

So the main system contribution that survives both workloads is:

- predictive model-state residency
- predictive prefetch
- shell-aware resource partitioning

while phase shifting should remain an optional optimization rather than a universal default.
