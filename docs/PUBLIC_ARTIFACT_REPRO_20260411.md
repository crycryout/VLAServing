# Public-Code / Artifact-Level Reproduction Status for Single-GPU Serving Methods

This note records what we were able to reproduce from public code, and what we observed when adapting the public methods to VLA-style workloads.

Related scripts and results:

- Public-code reproduction script:
  - [`src/repro_public_gpu_serving_artifacts.py`](../src/repro_public_gpu_serving_artifacts.py)
- Public-code reproduction results:
  - [`results/public_gpu_serving_artifacts_20260411.json`](../results/public_gpu_serving_artifacts_20260411.json)
- Unified workload reproduction:
  - [`src/bench_vla_single_gpu_methods.py`](../src/bench_vla_single_gpu_methods.py)
  - [`results/vla_single_gpu_methods_20260411.json`](../results/vla_single_gpu_methods_20260411.json)

## Scope

We distinguish two levels of reproduction:

1. `paper-faithful reproduction`
   - A unified evaluation driven by real local measurements of inference latency, shell memory footprint, H2D copy time, current Pi0.5 `25/50` control semantics, GR00T chunk dynamics, and action-exhaustion constraints.
   - This level was completed for `Clockwork-like`, `REEF-like`, `Paella-like`, `GPUlet-like`, and our `VLA-aware` scheduler on both `Pi0.5` and `GR00T N1.6`.

2. `artifact-level / public-code reproduction`
   - Running the actual released code when possible on this machine.
   - This is more constrained by hardware, drivers, runtimes, and the original software stack.

This document is about the second level.

## Environment

- GPU: `RTX 4090 24GB`
- Driver: `580.105.08`
- CUDA toolkit: `11.8`
- OS: `Ubuntu 22.04.1`

## GPUlet / glet (ATC'22)

Public repo:

- <https://github.com/casys-kaist/glet>

### What we reproduced

We reproduced the `standalone_scheduler` path, which is the closest artifact-level entry point for single-GPU static scheduling on this machine.

The local clone is at:

- `/root/autodl-tmp/glet`

The scheduler binary exists at:

- `/root/autodl-tmp/glet/bin/standalone_scheduler`

### Sanity run

The public scheduler works on a minimal single-task sanity case.

From [`results/public_gpu_serving_artifacts_20260411.json`](../results/public_gpu_serving_artifacts_20260411.json):

- `sanity_single_vgg16`
  - `exit_code = 0`
  - `output_exists = true`
  - `model_list = 0,100,0,vgg16,31,553.571472`

This confirms the public scheduler binary itself is usable on this host for a simple non-VLA task.

### VLA-adapted runs

We then generated scheduler inputs using our measured `Pi0.5` and `GR00T N1.6` batch curves, shell memory sizes, and VLA-style task sets.

Cases:

- `gr00t_two_model_gpulet`
  - two-model GR00T case
  - `exit_code = 124`
  - timed out after `30s`
- `gr00t_four_model_gpulet`
  - four-model GR00T case
  - `exit_code = 124`
  - timed out after `30s`
- `pi05_four_model_scaledx10_gpulet`
  - four-model Pi0.5 case with integer-rate scaling
  - `exit_code = 124`
  - timed out after `30s`

No `ModelList.txt` was produced in any of these VLA-adapted cases.

### Why this matters

This is already evidence that the public GPUlet scheduler is not a good fit for the VLA problem we care about:

- It expects an offline static schedule over integer request rates and SLOs.
- It does not model:
  - multi-finetuned-model shell residency,
  - predictive prefetch,
  - H2D bandwidth as a schedulable resource,
  - Pi0.5's legal replan window under the current `25/50` control semantics,
  - action-exhaustion hard deadlines.

Two additional mismatches are explicit in the public-code run:

1. `Pi0.5` request-rate mismatch
   - The public scheduler only accepts integer request rates.
   - Pi0.5 VLA request rates under chunked control semantics are fractional and model-dependent.
   - This means the public scheduler cannot faithfully encode the real Pi0.5 workload even before model-state management is considered.

2. Static search complexity under VLA-like profile inputs
   - Even coarse two-part cases like `[50, 100]` on one GPU did not finish within `30s`.
   - That does not prove the scheduler can never find an answer, but it does show that the public offline scheduler is not a practical runtime for fine-grained VLA control-serving.

## REEF (OSDI'22)

Public repos:

- <https://github.com/SJTU-IPADS/reef>
- <https://github.com/SJTU-IPADS/reef-artifacts/tree/osdi22-ae>

### Artifact-level status

We did not continue artifact-level reproduction on this machine because the official stack is fundamentally incompatible with the host.

Official artifact assumptions:

- AMD Radeon Instinct `MI50`
- `ROCm 4.3.0`
- customized `amdgpu` kernel driver
- customized `rocclr` and `hip`

Current host:

- NVIDIA `RTX 4090`
- CUDA, not ROCm
- no `amdgpu`
- no `hipcc`

### Conclusion

REEF artifact-level reproduction is not meaningful on this host.

This is not a minor dependency issue. The mismatch is at:

- hardware,
- kernel driver,
- and runtime stack.

For REEF, the right comparison on this machine remains the completed `paper-faithful reproduction` in [`results/vla_single_gpu_methods_20260411.json`](../results/vla_single_gpu_methods_20260411.json).

## Paella (SOSP'23)

Public repo:

- <https://github.com/MachineLearningSystem/23sosp-paella>

### Artifact-level status

Minimal artifact-level reproduction may be possible on this host, but not for `Pi0.5` / `GR00T` directly.

Blocking issues:

- missing dependencies in the current environment:
  - `Boost`
  - `spdlog`
  - `clang`
  - `tvm`
- Paella depends on a customized TVM stack:
  - `tvm-llis`
- Paella expects models compiled into LLIS/TVM jobs, not PyTorch/HF/torch.compile workloads.

### Conclusion

Paella's public code is not a direct path for running `Pi0.5` / `GR00T` VLA workloads.

The closest artifact-level next step would be:

- run Paella's own TVM CNN jobs on this machine,
- but keep `Pi0.5` / `GR00T` comparisons at the `paper-faithful reproduction` level.

## Bottom line

Artifact-level reproduction currently supports the following claim:

- `glet` public code can be built and its scheduler can run on a trivial single-task sanity case.
- Once adapted toward realistic VLA-style `Pi0.5` / `GR00T` multi-model inputs, the public scheduler becomes impractical and fails to produce a schedule within the experiment timeout.
- `REEF` artifact-level reproduction is blocked by hardware/runtime incompatibility.
- `Paella` artifact-level reproduction may be possible only for its native TVM jobs, not for the current VLA model stack.

This is consistent with the stronger measured conclusion from the unified evaluation:

- mainstream single-GPU serving abstractions are not sufficient for VLA because they do not jointly optimize:
  - future request predictability,
  - model-state residency and prefetch,
  - CPU-GPU bandwidth,
  - hard action-exhaustion deadlines,
  - and the control-semantic legal replan window.
