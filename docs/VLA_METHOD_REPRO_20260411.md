# VLA Single-GPU Serving Method Reproduction

This note summarizes a paper-faithful reproduction of representative single-GPU serving methods under a unified VLA workload.

## Goal

Test whether mainstream single-GPU serving abstractions remain suitable when all of the following are true at once:

- multiple fine-tuned models share one GPU
- requests follow workload-specific chunk control semantics
- request-to-result latency must be within `100ms`
- action chunks must never be exhausted before the next reply arrives

## Representative methods

- `Clockwork-like`
  - predictable reservation of compute service
  - no model-state-aware residency or prefetch
- `GPUlet-like`
  - single-GPU spatio-temporal partition with batch plus duty-cycle style execution
  - swap-aware coarse adaptation for multi-finetuned VLA models
- `REEF-like`
  - temporal-only preemptive scheduler
  - models still switch reactively
- `Paella-like`
  - low-latency reactive scheduler
  - model residency is still reactive rather than predictive
- `USHER-like`
  - interference-aware spatial multiplexing
  - fixed resident partitions, but cold-model replacement is still reactive
- `DistServe-like`
  - two-stage disaggregated pipeline
  - adds stage queues and duplicated model-state pressure on a single GPU
- `VLA-aware`
  - frequency-aware shell residency
  - next-request-aware prefetch
  - control-semantic request timing

## Workloads

### Pi0.5

- `30Hz -> 30hz_official_ft`
- `20Hz -> 20hz_quantiles`
- `10Hz -> 10hz_a_logits`
- `10Hz -> 10hz_b_autoh`

Constants:

- compiled inference about `43ms`
- full pinned H2D whole-model copy about `289ms`
- shell memory about `7.485 GiB`
- chunk size `50`
- control semantics:
  - horizon target in `{25, 50}`
  - legal replan window `[25, 50]`

### GR00T N1.6

- `30Hz -> 30hz_bridge`
- `20Hz -> 20hz_fractal`
- `10Hz -> 10hz_libero`
- `10Hz -> 10hz_rel30k`

Constants:

- compiled inference about `43.8ms`
- chunk size `16`
- shell memory:
  - `bridge/fractal` about `6.12 GiB`
  - `libero/rel30k` about `8.56 GiB`

## Request model

- `Pi0.5` uses the current `{25, 50}` control semantics with legal replan window `[25, 50]`.
- `GR00T N1.6` keeps its own chunk-level dynamic horizon process.
- A method must both:
  - return within `100ms`
  - and keep the chunk alive until the next reply arrives

## Main result

### Pi0.5

旧版 `Pi0.5` baseline 数字基于更早的非 `25/50` 语义，当前文档不再保留。

当前 `Pi0.5` 主线结论应以这两份结果为准：

- [`vla_gpu_virtualization_policy_20260412.json`](../results/vla_gpu_virtualization_policy_20260412.json)
- [`pi05_vla_serving_autoh25_50_phase_shift_20260413.json`](../results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json)

当前有效结论是：

- `Pi0.5` 使用 `{25, 50}` 控制语义和 `[25, 50]` legal replan window
- fixed-4 场景稳定在约 `43.21ms p95`
- admission 容量约为 `32.67`
- 主导收益来自 `legal replan window` 变宽，而不是更激进的 phase shift

### GR00T N1.6

`oracle_full_resident`

- mean SLA miss rate `0.4075`
- mean hard miss rate `0.0`
- mean fleet score `0.9063`
- mean min robot score `0.8839`
- mean latency p95 `140.34ms`
- memory `29.37 GiB`

`reef_like_temporal`

- mean SLA miss rate `1.0`
- mean hard miss rate `0.5195`
- mean latency p95 `1195.15ms`

`clockwork_like`

- mean SLA miss rate `0.9221`
- mean hard miss rate `0.0039`
- mean latency p95 `437.01ms`

`paella_like`

- mean SLA miss rate `0.9221`
- mean hard miss rate `0.0039`
- mean latency p95 `437.01ms`

`usher_like`

- mean SLA miss rate `1.0`
- mean hard miss rate `0.0020`
- mean latency p95 `520.65ms`

`distserve_like`

- mean SLA miss rate `1.0`
- mean hard miss rate `1.0`
- mean latency p95 `2689.92ms`

`vla_aware`

- mean SLA miss rate `0.4075`
- mean hard miss rate `0.0`
- mean fleet score `0.9063`
- mean min robot score `0.8839`
- mean latency p95 `140.34ms`
- memory `20.80 GiB`

Interpretation:

- Even the no-swap upper bound already misses the `100ms` SLA often.
- This means GR00T N1.6 with full fine-tuned model states is compute and horizon constrained under this workload, not just swap constrained.
- Traditional methods still perform much worse, but the gap to a feasible `100ms` system is not fully recoverable by better scheduling alone.
- USHER-like fixed spatial sharing again fails because it ignores request predictability and model-state movement.
- DistServe-like disaggregation is especially poor for GR00T because each request already sits near the compute limit; splitting it into extra stages only increases the critical path.

## Why mainstream methods fail for VLA

- `GPUlet-like`
  - its duty-cycle and batching windows exceed `100ms` once model residency and swap are accounted for
  - even coarse feasibility already fails under the single-GPU memory budget

- `Clockwork-like`
  - predictable compute reservation alone is insufficient
  - reactive cold-model loading still dominates latency

- `REEF-like`
  - kernel preemption does not solve model-state movement
  - with multiple full fine-tuned models it becomes swap bound

- `Paella-like`
  - reactive low-latency scheduling helps only when models are already resident
  - when one model is cold, the critical path is still swap plus inference
- `USHER-like`
  - spatial interference management assumes the main problem is co-running compute jobs
  - VLA instead becomes dominated by when model state is resident and when future requests should trigger prefetch
- `DistServe-like`
  - stage disaggregation helps LLM-style prefill/decode asymmetry, not short monolithic VLA forward passes
  - on one GPU it adds more queues and more reactive model-state movement

## Key conclusion

Traditional single-GPU serving methods are not directly suitable for VLA workloads because they optimize compute scheduling after a request arrives, while VLA requires:

- predictive model-state residency and prefetch before the request
- control-semantic hard deadlines
- workload-aware legal replan windows
- multi-model memory virtualization as a first-class resource

## Files

- benchmark driver: `src/bench_vla_single_gpu_methods.py`
- result JSON: `results/vla_single_gpu_methods_20260411.json`
