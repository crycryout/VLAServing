# Unified Chunked VLA Serving System

This document defines one serving architecture that covers both:

- `pi05` style families, where the critical issue is model-state movement and exact-delta residency/prefetch.
- `GR00T N1.6` style families, where the critical issue is phase locking and same-model batching under shared-prefix residency.

The goal is not to force every family through one runtime trick. The goal is to expose one general control plane and allow the data plane to specialize by model family.

## Core Claim

Any chunked-action VLA family can be served by the same high-level system if the runtime is parameterized by:

- chunk size,
- horizon process,
- request frequency,
- legal early-replan window,
- latency model,
- fine-tune memory relation.

The system should not be organized around a specific backbone or a specific checkpoint format. It should be organized around the temporal structure of chunked control.

## Shared Control Plane

Every supported family uses the same top-level components:

1. `Timeline Predictor`
   Predicts the next inference request time of every robot from the model's chunk horizon semantics.

2. `Admission Controller`
   Accepts a robot only if the predicted steady-state remains deadline-safe and fair.

3. `Memory State Manager`
   Owns resident model state on the GPU:
   - full shells,
   - shared prefixes,
   - exact delta pages,
   - decode/apply buffers.

4. `Compute Scheduler`
   Decides which requests run next:
   - single request,
   - same-model batch,
   - optional concurrent wave if it is actually beneficial.

5. `Phase Controller`
   Uses legal early replan to move future requests into better cohorts when the family supports that behavior.

## Backend A: ExactDeltaPrefetch

Use this when:

- full fine-tunes do not all fit as resident copies,
- exact or lossless deltas exist against a base shell,
- transfer plus decode plus apply dominates over compute.

This is the Pi05-like path.

### Runtime pattern

- keep a small number of active shells resident,
- keep compressed or exact delta pages resident by frequency,
- maintain future request times from AutoHorizon,
- prefetch and decode the next model state before the request reaches the shell,
- overlap model-state preparation with current compute when possible.

### What matters most

- residency fraction by model,
- future request predictability,
- H2D plus decode bandwidth,
- shell reuse.

Batching is not the first-order win here.

## Backend B: SharedPrefixPhaseBatch

Use this when:

- all fine-tunes can stay resident, or
- they share a resident prefix and only differ in lightweight task-specific state,
- same-model batching has a favorable latency curve,
- the model supports legal early replan inside the chunk window.

This is the GR00T N1.6-like path.

### Runtime pattern

- keep all fine-tuned variants resident through shared-prefix storage,
- group requests by model,
- phase-lock same-model robots so future requests align,
- batch same-model requests,
- use fairness-aware admission so greedy batching does not bias the fleet.

### What matters most

- same-model batch curve,
- phase correction budget,
- per-model quota fairness,
- avoiding overuse of MPS.

MPS is optional. It is not the default unless it increases stable capacity in measured runs.

## Backend Selection Rule

For a new chunked-action family:

1. Measure single-request latency.
2. Measure same-model batch latency, if batching is supported.
3. Measure full-shell memory.
4. Measure exact-delta or shared-prefix memory, if available.
5. Characterize the horizon process and legal early-replan window.

Then choose:

- `SharedPrefixPhaseBatch` if same-model batch plus resident prefix is the main gain.
- `ExactDeltaPrefetch` if memory movement is the main gain.
- `HybridChunkedRuntime` if the family is mixed and needs per-model backend selection.

## Current Default Instantiations

### Pi05

- backend: `ExactDeltaPrefetch`
- scheduler: single active shell plus predictive prefetch
- phase control: weak or optional
- admission: frequency-aware predictive admission
- validated reference:
  [pi05_four_model_residency_prefetch_system_20260406.json](../results/pi05_four_model_residency_prefetch_system_20260406.json)

### GR00T N1.6

- backend: `SharedPrefixPhaseBatch`
- scheduler: same-model phase-lock batch
- phase correction budget: `4`
- admission: quota-fair admission
- validated references:
  [gr00t_shared_prefix_phase_lock_batch_mps_20260412.json](../results/gr00t_shared_prefix_phase_lock_batch_mps_20260412.json)
  [gr00t_batch_only_fair_admission_20260412.json](../results/gr00t_batch_only_fair_admission_20260412.json)

## Why This Generalizes

The system does not assume:

- a specific transformer backbone,
- a specific tokenizer or action head,
- a specific checkpoint format,
- a specific vendor runtime.

It only assumes:

- chunked action outputs,
- predictable re-request structure,
- measurable latency behavior,
- measurable memory relation among fine-tunes.

That is the right abstraction boundary for a general VLA serving system.
