# VLA Single-GPU Serving Method Reproduction

This note summarizes a paper-faithful reproduction of representative single-GPU serving methods under a unified VLA workload.

## Goal

Test whether mainstream single-GPU serving abstractions remain suitable when all of the following are true at once:

- multiple fine-tuned models share one GPU
- requests follow chunk-level AutoHorizon dynamics
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
  - AutoHorizon-driven request timing

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

- Each chunk draws a new AutoHorizon.
- A request is issued at:

  `request_time = chunk_start + horizon * action_period - 100ms`

- So a method must both:
  - return within `100ms`
  - and keep the chunk alive until the next reply arrives

## Main result

### Pi0.5

`oracle_full_resident`

- all 4 models resident
- mean SLA miss rate `0.0497`
- mean hard miss rate `0.0082`
- mean fleet score `0.9920`
- mean min robot score `0.9847`
- mean latency p95 `104.92ms`
- memory `29.94 GiB`

`reef_like_temporal`

- mean SLA miss rate `0.9132`
- mean hard miss rate `0.3799`
- mean latency p95 `752.67ms`

`clockwork_like`

- mean SLA miss rate `0.2425`
- mean hard miss rate `0.1538`
- mean latency p95 `364.84ms`
- memory `22.46 GiB`

`paella_like`

- mean SLA miss rate `0.2425`
- mean hard miss rate `0.1538`
- mean latency p95 `364.84ms`
- memory `22.46 GiB`

`usher_like`

- mean SLA miss rate `1.0`
- mean hard miss rate `0.3850`
- mean latency p95 `471.71ms`
- memory `22.46 GiB`

`distserve_like`

- mean SLA miss rate `0.9667`
- mean hard miss rate `0.5151`
- mean latency p95 `1980.08ms`
- memory `14.97 GiB`

`vla_aware`

- mean SLA miss rate `0.0497`
- mean hard miss rate `0.0082`
- mean fleet score `0.9920`
- mean min robot score `0.9847`
- mean latency p95 `104.92ms`
- memory `22.46 GiB`

Interpretation:

- The VLA-aware system matches the no-swap upper bound behavior while using only `22.46 GiB`, not `29.94 GiB`.
- Clockwork-like and Paella-like reactive systems still fail because one model remains cold and pays reactive swap on the critical path.
- REEF-like temporal preemption fails worst because preemption cannot remove model-switch cost.
- USHER-like spatial partitioning still fails because fixed resident partitions do not solve predictive model-state placement.
- DistServe-like stage disaggregation performs worst among non-REEF baselines because VLA inference is not naturally prefill/decode separable, so the extra stage queues and duplicated cold-model swaps dominate latency.

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
- AutoHorizon-aware soft scheduling targets
- multi-model memory virtualization as a first-class resource

## Files

- benchmark driver: `src/bench_vla_single_gpu_methods.py`
- result JSON: `results/vla_single_gpu_methods_20260411.json`
