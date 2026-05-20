# Predictable VLA MegaKernel: Top-Conference Idea, Roadmap, and Evaluation Plan

Date: 2026-05-20

This document is a paper-idea-level summary of the VLA Mega Kernel direction
discussed in this project. It is intended to capture the full system insight,
top-conference framing, implementation roadmap, and evaluation plan.

The central message is:

```text
VLA serving is not ordinary random-arrival neural serving. Closed-loop robot
control exposes predictable action-exhaustion windows, phase structure, and
deadline semantics. A serving system can exploit this predictability to move the
latency-critical path from host-driven queueing into a GPU-resident,
phase-aware, preallocated, prefetchable MPK-style data plane.
```

This document does not claim that a full high-performance `VLM + DiT` GR00T
Mega Kernel already exists. The current evidence supports the system direction
and identifies the required engineering steps.

## 1. Paper Thesis

The proposed paper should be framed around a new workload class:

```text
Predictable closed-loop VLA serving.
```

Unlike LLM serving, where requests arrive unpredictably and are usually treated
as queue items, VLA serving is driven by robot action consumption. Once a robot
receives an action chunk, the system can predict when that robot will need the
next chunk. The system can also shape future requests using phase control,
admission, horizon policy, and fairness constraints.

The paper thesis:

```text
VLA request predictability enables a GPU-resident persistent data plane that
uses fixed phases, fixed buffers, fixed shape buckets, request/completion rings,
stage credits, and MPK-style task fusion to reduce synchronization overhead,
batch waiting, kernel boundary gaps, and resource contention.
```

The shortest version:

```text
Turn VLA serving from queue-first host scheduling into lease-first GPU-resident
phase scheduling.
```

## 2. Why This Can Be a Top-Conference Systems Idea

The work is not just "make one kernel faster." The stronger systems idea is:

1. Define VLA serving as a predictable real-time neural workload.
2. Show why existing GPU serving abstractions are mismatched:
   - queue-first batching,
   - coarse MPS/MIG sharing,
   - host-driven CUDA Graph replay,
   - model-server style request scheduling.
3. Propose a new GPU data-plane abstraction:
   - phase-aware lanes,
   - request leases,
   - horizon/deadline semantics,
   - preallocated fixed buffers,
   - GPU-side scheduling,
   - stage-level resource credits.
4. Show that the abstraction improves both:
   - single-inference latency potential,
   - fleet-level robot admission, deadline, and fairness.
5. Use GR00T N1.6 as the primary case study and Pi0.5/Pi05-style residency as
   supporting evidence for broader VLA serving.

The paper should target a systems venue angle:

```text
The contribution is a VLA-aware GPU serving architecture, not only a custom CUDA
kernel.
```

### 2.1 Comparison With Traditional Serving Systems

Traditional serving systems usually optimize one of three abstractions:

1. request/response model serving,
2. queue-first dynamic batching,
3. coarse GPU sharing or graph replay.

Those abstractions miss the most useful property of VLA serving: the system can
predict future action-refresh windows and can shape phases before requests hit
the execution path.

| Traditional serving mechanism | What it optimizes | Why it is insufficient for VLA | Proposed VLA optimization |
|---|---|---|---|
| FIFO / request-response model serving | simple online latency | ignores action exhaustion, horizon penalty, and robot phase | lease/deadline-aware request model |
| dynamic batching | throughput by waiting for more requests | batch wait consumes action budget and can bias toward high-frequency robots | phase-aware batch1 lanes and deadline admission |
| continuous batching in LLM serving | token-level throughput and KV reuse | VLA action chunks have hard refresh windows, not just decode token queues | action-horizon scheduling and phase movement |
| CUDA Graph replay | lower host launch overhead for fixed graphs | replay is still host-triggered and does not manage multi-robot phase/resource contention | GPU-resident persistent request data plane |
| TensorRT / optimized inference engine | faster single-model kernels | does not solve fleet-level admission, fairness, or phase waiting | combine optimized kernels with VLA-aware serving control |
| MPS / multi-stream overlap | coarse concurrency | can create stage backlog, prefill overrun, and uncontrolled resource contention | per-stage credit and phase-local deadlines |
| MIG / spatial partitioning | hard resource isolation | partitions capacity statically and ignores temporal phase slack | elastic phase-aware lanes sharing one model |
| generic pipeline parallelism | overlap stages | ordinary pipeline does not know robot deadlines or success penalty | horizon-aware VLM/DiT stage scheduling |
| CPU/GPU co-processing | use more heterogeneous resources | dense VLM/DiT CPU offload is usually too slow; sync can dominate | CPU as low-rate control/pre/post plane, GPU as persistent data plane |

The key mismatch:

```text
Traditional serving treats requests as arrivals to a queue.
VLA serving should treat future action refreshes as schedulable leases.
```

### 2.2 Main Optimization Points Over Traditional Serving

The proposed system optimizes at three layers.

Single-inference layer:

| Optimization | Traditional behavior | VLA MegaKernel behavior | Expected benefit |
|---|---|---|---|
| host launch removal | CPU launches many kernels or graph replays per request | CPU writes descriptor; GPU persistent data plane runs tasks | lower request-path CPU overhead |
| graph/operator boundary reduction | VLM/DiT executes many GEMM/norm/copy/shape kernels | MPK/Mirage fuses task boundaries where practical | lower timeline gaps and HBM round trips |
| preallocation | tensor allocation and dynamic addresses appear in request path | fixed per-lane arenas and pointer-stable buffers | less allocator overhead, easier capture |
| fixed-shape buckets | dynamic shape handling and graph breaks | pad/mask into finite shape classes | more graph/MPK reuse |
| prefetch/load | data and weights loaded reactively | prefetch static weights, stage buffers, and next-lane scratch by phase | lower memory wait and better cache locality |

Serving-system layer:

| Optimization | Traditional behavior | VLA MegaKernel behavior | Expected benefit |
|---|---|---|---|
| phase-aware admission | admit by queue capacity or average latency | admit by future action-exhaustion windows | higher stable robot count |
| five-phase lanes | wait for group batch | frequent batch1 service slots with full active latency charged | less queue wait and better deadline behavior |
| fairness-aware scheduling | high-frequency robots can dominate admission | acceptance constrained across 10Hz/20Hz/30Hz classes | lower accept-rate gap |
| horizon penalty awareness | latency objective only | serving time tied to success penalty | better fleet score and task success |
| queue-to-lease conversion | reactive queueing | pre-scheduled leases with phase movement | fewer surprise overloads |

Resource-control layer:

| Optimization | Traditional behavior | VLA MegaKernel behavior | Expected benefit |
|---|---|---|---|
| stage credit | MPS/streams let stages compete implicitly | explicit per-stage concurrency limits | controlled SM/HBM/L2/shared/register contention |
| phase release | burst arrivals collide in same stage | lane starts are phase-shifted | fewer same-stage collisions |
| GPU-side scheduling | CPU orders streams and launches | persistent kernel selects ready lane/stage | lower host scheduling overhead |
| model residency | serving system treats requests independently | one shared GR00T model with fixed buffers | less repeated setup and better cache/layout reuse |
| request/completion rings | CPU synchronizes per request or stage | lightweight descriptor/completion protocol | fewer CPU-GPU sync points |

The cleanest top-level comparison:

```text
Dynamic batching improves throughput by waiting.
Predictable VLA MegaKernel improves throughput and deadlines by scheduling
future leases before they become urgent.
```

## 3. Core Insight

The key insight is that VLA serving exposes useful predictability at several
levels.

| Level | Predictable property | Optimization enabled |
|---|---|---|
| robot control | action consumption rate | next request time window |
| horizon | legal refresh window and penalty | phase movement and deadline-aware scheduling |
| model path | fixed VLM/DiT shape buckets | CUDA Graph / MPK capture, static buffers |
| pipeline | fixed stage order | prefetch and stage handoff |
| fleet | repeated 10Hz/20Hz/30Hz cohorts | phase-aware admission and fairness |
| memory | fixed model and scratch layout | preallocation, residency, pointer-stable descriptors |

The important wording is:

```text
VLA requests are not perfectly deterministic, but they have predictable windows.
The system can use those windows to make execution, memory, and synchronization
plans much more static than in ordinary request serving.
```

This makes three classes of optimization possible:

1. Single-inference optimization:
   - fewer launches,
   - fewer graph/operator boundaries,
   - lower CPU/GPU sync overhead,
   - less intermediate tensor materialization,
   - more fusion and tile prefetch.
2. Serving-system optimization:
   - lower batch waiting,
   - higher stable robot count,
   - lower deadline miss / horizon penalty,
   - fairer acceptance across control frequencies.
3. Resource-control optimization:
   - phase-separated concurrent lanes,
   - per-stage SM/HBM/L2/shared/register credits,
   - explicit collision control inside one persistent GPU data plane.

## 4. Problem Setting

The main workload is GR00T N1.6 VLA serving:

- multiple robots share one GPU,
- robots consume action chunks at different frequencies,
- request deadlines are tied to action exhaustion,
- requests can be shifted within a legal window,
- serving too late can reduce success,
- the same model is shared by all robots.

Current Admission/Horizon policy used in this workspace:

| Horizon case | Serving rule |
|---|---|
| `horizon <= 8` | only consumed action `8` is penalty-free; serving later inside `[8, 16]` is allowed but increasingly penalized |
| `horizon > 8` | serving anywhere in `[8, 16]` is legal and penalty-free |

This matters because ordinary batching waits for enough requests, but VLA
waiting can consume the robot's action budget and reduce success.

## 5. Baseline Position

The strongest current practical GR00T serving baseline is:

```text
fused_conservative = VLM(batch1) + DiT microbatch / step CUDA Graph
```

Earlier admission experiments reported a strong baseline around:

| Metric | Value |
|---|---:|
| `mean_final_robot_count` | `28.50` |
| `mean_p95_ms` | `72.85 ms` |

This baseline is important because optimistic MPS/multistage pipeline models can
look competitive, but conservative request-path models are weaker. The paper
must compare against this style of strong baseline, not only against eager
PyTorch or naive FIFO serving.

For official single-request GR00T N1.6 timing, public README-style numbers have
reported RTX 4090 `torch.compile` around:

| Component | Public reported RTX 4090 torch.compile timing |
|---|---:|
| data processing | `2 ms` |
| backbone | `25 ms` |
| action head | `17 ms` |
| E2E | `44 ms` |

Local fresh-stack profiling in this workspace measured a slower dataset-backed
profiling path, around `54-56 ms` E2E. These two scopes must not be conflated.
All comparisons need the same model, input shape, dtype, inference mode, and
measurement boundary.

## 6. Proposed System Architecture

The design has two planes.

```text
VLA-aware control plane:
  admission
  horizon and phase policy
  fairness
  safety checks
  CPU preprocessing / postprocessing
  edge and network decisions

GPU-resident MPK data plane:
  request/completion rings
  fixed-phase inference lanes
  preallocated per-lane buffers
  VLM/DiT task execution
  per-stage resource credits
  GPU-side completion reporting
```

The CPU should not disappear. It should leave the request-critical execution
path.

Traditional host-driven path:

```text
CPU prepares tensors
CPU launches VLM kernels
CPU hits graph or stream boundary
CPU launches DiT kernels
CPU waits or copies result
CPU schedules the next request
```

Target data-plane path:

```text
CPU writes a lightweight descriptor:
  robot_id
  phase
  horizon
  deadline
  input pointer
  output pointer
  valid bit

GPU persistent scheduler:
  consumes descriptor
  selects lane and stage
  prefetches inputs / weights / scratch
  runs VLM and DiT tasks
  writes action output
  writes completion flag
```

This is "sync-light" rather than literally sync-free:

```text
The goal is to remove per-request and per-stage CPU/GPU synchronization from
the latency-critical path.
```

## 7. Five-Phase Lane Abstraction

The near-term serving abstraction is a five-phase batch1 lane design:

```text
one persistent outer Mega Kernel
  lane 0 starts at phase 0
  lane 1 starts at phase L / 5
  lane 2 starts at phase 2L / 5
  lane 3 starts at phase 3L / 5
  lane 4 starts at phase 4L / 5

Each lane is internally serial.
Each request still has active latency L.
The system receives a new service slot every L / 5.
```

Important clarification:

```text
The slot interval is L / 5. The latency of one request is still L.
```

This matters because a fair evaluation must charge each five-phase request the
full active latency of a real concurrent lane. The five-phase result is still
useful even when each lane is charged the latency of traditional batch4 or
batch5 inference.

## 8. Stage Credit and Resource Control

Five concurrent lanes can create resource contention:

- SM contention,
- L2 contention,
- HBM bandwidth contention,
- shared-memory pressure,
- register pressure,
- scheduler contention,
- stream or launch ordering overhead if implemented outside a persistent kernel.

The proposed runtime handles this with per-stage credit:

```text
credit[stage] = maximum number of lanes allowed to occupy that stage class
```

Examples:

- compute-heavy GEMM/FFN stage may have one credit limit,
- memory-heavy copy/norm/index stage may have another,
- VLM attention and DiT attention may need separate limits,
- burst arrivals should be converted into phase releases when possible.

Credit is not a universal win. If credit is too strict, throughput collapses.
The runtime must tune credit by measured resource pressure.

## 9. MPK / Mirage Role

The MPK/Mirage line should be positioned as the compute data-plane layer:

```text
VLA control plane decides when and what to run.
MPK data plane decides how to run it inside the GPU with minimal host overhead
and cross-operator fusion.
```

MPK can help at three levels:

1. Intra-stage:
   - fuse linear/norm/activation/residual,
   - reduce small-kernel fragmentation,
   - improve tile prefetch and epilogue fusion.
2. Inter-stage:
   - reduce VLM-to-DiT handoff overhead,
   - keep intermediate buffers in stable locations,
   - avoid CPU stage boundaries.
3. Inter-request:
   - run multiple phase-shifted lanes inside one persistent kernel,
   - enforce stage credit inside the GPU,
   - avoid per-request host scheduling.

The practical implementation order should not start with full VLM mega-kernel.
The current best order is:

```text
DiT step / mixed-step tasks
-> VLM postprefill subgraph tasks
-> five-phase persistent data plane
-> broader VLM integration
```

## 10. Current Evidence

### 10.1 VLM Has Removable Data-Plane Gap

Source:

- `results/gr00t_vlm_mpk_dataplane_proxy_20260510.json`

| Metric | Value |
|---|---:|
| VLM uncaptured fixed-shape CUDA p50 | `26.78 ms` |
| VLM graph replay CUDA p50 | `10.82 ms` |
| speedup | `2.47x` |
| math kernel sum ratio | `0.967` |
| timeline gap reduction | `25.47 ms` |

Interpretation:

- The VLM path has substantial launch/boundary/timeline gap.
- Graph replay keeps essentially the same math work but reduces wall time.
- This is strong evidence for MPK-style data-plane potential on VLM.
- It is not yet a full VLM MPK mega-kernel.

### 10.2 Single-Request MPK Potential Proxy

Source:

- `results/gr00t_official_mpk_potential_graphproxy_20260510.json`

| Metric | Value |
|---|---:|
| local-harness estimated E2E p50 | `53.63 ms` |
| model CUDA kernel sum | `23.64 ms` |
| idealized E2E lower bound | `29.91 ms` |
| potential speedup | `1.79x` |

Interpretation:

- This is an upper-bound/proxy for removable timeline gap.
- It should not be reported as implemented MPK speedup.
- It motivates reducing boundary/gap overhead while preserving math efficiency.

### 10.3 Official Clean Stack Launch / Sync Profiling

Sources:

- `results/official_clean_groot_n16_3b_torchcompile_launch_sync_20260520.json`
- `results/official_clean_groot_n16_noncompute_overhead_breakdown_20260520.json`
- `results/official_clean_groot_n16_operator_inventory_20260520.json`

Fresh official checkout:

| Item | Value |
|---|---|
| checkout | `/root/autodl-tmp/Isaac-GR00T-official-clean-20260520` |
| branch | `n1.6-release` |
| commit | `ead52833afbbf4243f8cd5e7664f48a94de03b19` |
| model | `nvidia/GR00T-N1.6-3B` |
| GPU | RTX 4090 |
| torch | `2.7.1+cu126` |

Launch/sync breakdown:

| target | host enqueue | sync wait | wall sync | CUDA event | CPU CUDA launch/runtime |
|---|---:|---:|---:|---:|---:|
| VLM/backbone | `31.46 ms` | `0.64 ms` | `32.10 ms` | `31.94 ms` | `7.07 ms` |
| DiT action head | `7.13 ms` | `9.92 ms` | `17.05 ms` | `16.09 ms` | `3.64 ms` |
| full model GPU-only | `47.05 ms` | `8.84 ms` | `56.08 ms` | `55.47 ms` | `10.30 ms` |

Interpretation:

- `sync wait` is not pure sync overhead; it includes waiting for remaining GPU
  work at the boundary.
- CPU launch/runtime work overlaps with GPU compute.
- CUDA Graph can likely remove a few to about ten milliseconds of launch/runtime
  exposure in this local harness.
- MPK targets deeper overhead than CUDA Graph: graph/operator boundaries,
  intermediate materialization, and in-GPU scheduling.

Non-compute proxy:

| target | CUDA event | profiler kernel union | estimated non-kernel gap |
|---|---:|---:|---:|
| backbone prepared | `26.653 ms` | `8.886 ms` | `~17.58 ms` |
| action head prepared | `16.931 ms` | `14.835 ms` | `~2.05 ms` |
| full model prepared | `49.836 ms` | `23.303 ms` | `~26.29 ms` |

This is torch-profiler proxy evidence, not Nsight Compute roofline evidence.

### 10.4 Operator Inventory

Full model CUDA kernel class counts:

| kernel class | full model count | full model time |
|---|---:|---:|
| GEMM/MM | 1286 | `20.191 ms` |
| attention | 32 | `0.043 ms` |
| norm | 318 | `0.722 ms` |
| copy/cast | 510 | `0.920 ms` |
| elementwise | 398 | `0.483 ms` |
| activation | 48 | `0.118 ms` |
| shape/index | 113 | `0.387 ms` |
| memcpy | 108 | `0.113 ms` |
| memset | 61 | `0.029 ms` |
| other | 242 | `0.560 ms` |

Implication:

- Dense math is dominated by GEMM/MM.
- The path also contains many copy/cast, norm, elementwise, and shape/index
  kernels.
- These low-arithmetic kernels create fragmentation and boundary overhead, even
  if individual kernels are not large.

Compute/memory classification:

| Operator type | Expected bottleneck |
|---|---|
| GEMM / FFN / QKV / projections | compute-heavy, Tensor Core bound when shape is large enough |
| FlashAttention / SDPA / FMHA | mixed, shape-dependent |
| LayerNorm / RMSNorm / AdaLayerNorm | memory-bound |
| residual / mask / elementwise | memory-bound or launch-bound |
| copy / cast / contiguous / cat | memory-bound or dispatcher-bound |
| slice / index / gather / padding | memory/L2-bound |
| view / reshape / transpose | metadata / dispatcher-bound |

Final roofline classification requires Nsight Compute counters.

### 10.5 Synthetic MPK Admission Profile

Sources:

- `results/gr00t_mpk_synthetic_admission_profile_20260511.json`
- `docs/GR00T_MPK_SYNTHETIC_SERVING_PROFILE_20260511.md`

| metric | no-MPK fused | batched synthetic MPK | five-phase MPK lanes |
|---|---:|---:|---:|
| stable robot count | `12.67` | `19.00` | `32.50` |
| request p95 | `72.85 ms` | `53.16 ms` | `53.16 ms` |
| queued request p95 | `842.87 ms` | `1040.41 ms` | `705.37 ms` |
| queue wait p95 | `788.78 ms` | `996.95 ms` | `652.21 ms` |
| mean batch size | `6.19` | `7.37` | `1.00` |
| accept-rate gap | `0.203` | `0.207` | `0.0075` |
| lane utilization | n/a | n/a | `0.879` |

Interpretation:

- Faster batched inference alone can still create queueing because it admits
  more robots but waits for batches.
- Five-phase lanes reduce batch waiting and improve fairness.
- This is simulator evidence, not a real full MPK benchmark.

### 10.6 Equal-Latency Five-Phase Ablation

Source:

- `results/gr00t_phase5_equal_latency_admission_20260511.json`

This ablation removes the assumption that MPK is faster. Each five-phase request
is charged a traditional batch latency.

| mode | stable robots | request p95 | queued p95 | queue wait p95 | accept-rate gap |
|---|---:|---:|---:|---:|---:|
| no-MPK fused | `12.67` | `72.85 ms` | `842.87 ms` | `788.78 ms` | `0.203` |
| five-phase, latency=batch4 | `30.67` | `56.57 ms` | `719.34 ms` | `662.77 ms` | `0.0046` |
| five-phase, latency=batch5 | `29.50` | `59.71 ms` | `723.44 ms` | `663.74 ms` | `0.0607` |
| five-phase, latency=batch8 | `22.33` | `72.85 ms` | `743.11 ms` | `670.25 ms` | `0.0719` |

Key conclusion:

```text
Even when five-phase lanes are charged batch4 or batch5 active latency, the
serving shape remains better than ordinary group batching in stable capacity,
queue wait, and fairness.
```

This is the strongest current serving-system insight.

### 10.7 Persistent Scheduler MVP

Sources:

- `results/gr00t_phase5_cyclic_fixedlane_mpk_mvp_20260511.json`
- `results/gr00t_phase5_persistent_credit_mpk_mvp_20260511.json`
- `docs/GR00T_PHASE5_CYCLIC_FIXEDLANE_MPK_MVP_20260511.md`
- `docs/GR00T_PHASE5_PERSISTENT_CREDIT_MPK_MVP_20260511.md`

Representative result:

| mode | steady E2E p50 | active E2E p50 | queue p50 | max same-stage |
|---|---:|---:|---:|---:|
| solo | `3.186 ms` | `3.186 ms` | `0.000 ms` | `1 / 1` |
| cyclic burst no credit | `5.477 ms` | `3.357 ms` | `2.111 ms` | `5 / 5` |
| cyclic burst credit=3 | `4.299 ms` | `3.271 ms` | `1.030 ms` | `3 / 3` |
| cyclic phase no credit | `3.649 ms` | `3.270 ms` | `0.391 ms` | `4 / 4` |
| cyclic phase credit=3 | `3.794 ms` | `3.273 ms` | `0.525 ms` | `3 / 3` |
| cyclic phase credit=2 | `7.124 ms` | `3.369 ms` | `3.455 ms` | `2 / 2` |

Interpretation:

- Phase release reduces queueing and same-stage collision versus burst release.
- Credit can help collision-heavy cases.
- Too-strict credit damages throughput.
- This proves the control mechanism concept, not the final GR00T compute path.

### 10.8 Mirage / MPK Compilation Status

| Item | Current status |
|---|---|
| DiT attention Mirage compile | self/cross padded attention compile and run |
| DiT MPK attention bridge | task-variant bridge correctness close |
| VLM postprefill attention core | Qwen3 GQA shape compile started |
| VLM postprefill block skeleton | one decoder-block skeleton compile started |
| DiT hand-MPK single step | functional but far too slow |

Critical DiT result:

| path | latency |
|---|---:|
| official step | `17.88 ms` |
| CUDA Graph step | `4.73 ms` |
| current hand-MPK step | `190.07 ms` |

Conclusion:

```text
Current MPK plumbing works, but current hand-written task bodies are not the
performance path. The project needs shape-specific high-performance
linear/attention/FFN tasks, preferably Mirage/MPK-compiled or CUTLASS-style.
```

### 10.9 CPU-GPU Co-Processing Evidence

Sources:

- `results/gr00t_cpu_gpu_coprocessing_profile_iter8_20260520.json`
- `results/gr00t_cpu_gpu_coprocessing_dense_proxy_20260520.json`
- `results/gr00t_cpu_gpu_coprocessing_admission_proxy_20260520.json`

CPU/GPU overlap profile:

| Probe | Result |
|---|---:|
| CPU preprocessing p50 | `4.74 ms` |
| GPU model CUDA-event p50 | `46.05 ms` |
| sequential preprocess + GPU throughput | `19.88 req/s` |
| overlap GPU current + prepare next throughput | `22.17 req/s` |
| sequential per-request p50 | `50.26 ms` |
| overlapped steady iteration p50 | `44.78 ms` |

Dense CPU offload proxy:

| Proxy op | CPU bf16 p50 | GPU bf16 p50 | CPU/GPU ratio |
|---|---:|---:|---:|
| VLM postprefill dense `203x2048x2048` | `0.49 ms` | `0.027 ms` | `18.4x` |
| larger-token dense `1024x2048x2048` | `2.30 ms` | `0.070 ms` | `33.1x` |
| DiT small dense `16x1024x4096` | `0.19 ms` | `0.022 ms` | `8.9x` |
| DiT medium dense `64x1024x4096` | `0.26 ms` | `0.024 ms` | `11.1x` |

Conclusion:

- KTransformers-style CPU/GPU co-processing is useful inspiration.
- Current GR00T VLA dense VLM/DiT math should not be naively offloaded to CPU.
- CPU/edge should handle preprocessing, postprocessing, admission, safety,
  request rings, fallback, and overlap.

## 11. Research Questions

The evaluation should answer these questions.

| RQ | Question |
|---|---|
| RQ1 | Does VLA serving expose enough predictability to outperform queue-first batching? |
| RQ2 | Can five-phase lanes improve capacity and fairness even without faster per-request kernels? |
| RQ3 | How much CPU launch/sync and graph/operator boundary overhead exists in official GR00T N1.6? |
| RQ4 | Can MPK/Mirage-style data plane preserve math efficiency while reducing boundary/gap overhead? |
| RQ5 | Can a persistent GPU scheduler enforce phase release and stage credit without host stream ordering? |
| RQ6 | How much resource contention appears under concurrent lanes, and can credit control mitigate it? |
| RQ7 | Does CPU/edge co-processing help by overlap/control, and where does dense CPU offload fail? |
| RQ8 | Do latency improvements translate into closed-loop robot success and fewer horizon penalties? |

## 12. Evaluation Plan

### 12.1 Baselines

The paper should compare against:

| Baseline | Purpose |
|---|---|
| official GR00T N1.6 `torch.compile` | strongest standard PyTorch reference |
| CUDA Graph VLM/DiT subgraphs | launch-overhead baseline |
| fused conservative serving | current strongest practical serving baseline |
| group batch serving | queue-first batching baseline |
| DiT microbatch / step graph | current best DiT-side runtime line |
| MPS multistage serving | concurrent-stage baseline |
| naive prefill/LLM/DiT pipeline | shows backlog/handoff failure |
| synthetic equal-latency five-phase | isolates serving-shape value |
| persistent dummy scheduler | isolates GPU-side scheduling mechanism |
| CPU/GPU overlap baseline | heterogeneous overlap baseline |
| TensorRT path if available | vendor-optimized production baseline |

### 12.2 Workloads

Primary workload:

| Workload item | Value |
|---|---|
| model | GR00T N1.6 3B |
| robot embodiment | GR1 / same fine-tuned policy when applicable |
| action chunk | `16` |
| denoising steps | `4` |
| robot frequencies | 10Hz, 20Hz, 30Hz |
| horizon floor | `8` |
| horizon policy | penalty for `horizon <= 8` late service, no penalty for `horizon > 8` |
| input shape | fixed or finite-shape buckets |

Stress workloads:

- balanced 10Hz/20Hz/30Hz mix,
- 30Hz-heavy mix,
- 10Hz-heavy mix,
- bursty arrivals,
- jittered sensor arrival,
- high robot count until admission fails,
- edge/network delay variants.

### 12.3 Metrics

Single-inference metrics:

- data processing p50/p95,
- VLM p50/p95,
- DiT p50/p95,
- E2E p50/p95/p99,
- CUDA event time,
- CPU launch/runtime time,
- kernel count,
- CUDA Graph replay time,
- estimated boundary/gap time,
- numerical error versus official output.

Serving metrics:

- stable robot count,
- request p50/p95/p99,
- queued request p50/p95/p99,
- queue wait p50/p95/p99,
- deadline miss ratio,
- horizon penalty,
- fleet score,
- min robot score,
- accept-rate gap across frequencies,
- final accepted mix,
- lane utilization,
- mean batch size,
- reply-over count.

Resource metrics:

- SM throughput,
- Tensor Core utilization,
- DRAM throughput,
- L2 throughput and hit rate,
- achieved occupancy,
- register pressure,
- shared-memory pressure,
- warp stall breakdown,
- same-stage lane collision count,
- per-stage credit utilization.

CPU/edge metrics:

- CPU preprocessing latency,
- CPU postprocessing latency,
- request descriptor submit-to-start latency,
- completion-to-CPU-visible latency,
- CPU utilization,
- network p95/p99,
- H2D/D2H transfer time,
- edge preprocessing speed.

### 12.4 Evaluation Matrix

| Experiment | What it proves | Required artifact |
|---|---|---|
| official clean profiling | actual launch/sync/operator structure | torch profiler + Nsight Systems |
| VLM graph replay | removable VLM data-plane gap | CUDA event + profiler |
| DiT step graph | DiT lower-level baseline | CUDA Graph timing |
| operator inventory | target kernels and fragmentation | profiler operator table |
| Nsight Compute roofline | compute/memory classification | SM/HBM/L2/occupancy counters |
| equal-latency five-phase | serving-shape benefit without faster kernels | admission simulator |
| phase vs burst persistent MVP | phase release reduces collision | persistent dummy kernel trace |
| credit sweep | credit helps only when not over-throttled | persistent dummy kernel trace |
| request ring | CPU/GPU sync path is lightweight | ring microbenchmark |
| DiT task integration | real compute task feasibility | correctness + latency |
| VLM postprefill integration | VLM-side MPK value | correctness + latency |
| full five-phase replay | end-to-end serving benefit | runtime replay |
| CPU/edge overlap | heterogeneous value without dense offload | overlap and admission proxy |
| real rollout / success | system metrics map to task success | robot/sim evaluation |

### 12.5 Go / No-Go Criteria

The project should continue toward full MPK only if these hold:

| Gate | Required result |
|---|---|
| G1: boundary gap | official/VLM traces show enough removable overhead to matter |
| G2: serving shape | five-phase equal-latency remains better than group batching |
| G3: persistent scheduler | GPU-side phase release works without per-request host launches |
| G4: credit control | credit reduces collision without large throughput loss |
| G5: subgraph correctness | DiT/VLM tasks match official outputs within acceptable error |
| G6: subgraph performance | MPK/Mirage task is competitive with CUDA Graph / compiled baseline |
| G7: closed-loop value | lower queue/deadline penalty improves robot-level success or fleet score |

## 13. Implementation Roadmap

### Stage 0: Freeze Baselines

Deliverables:

- official clean GR00T N1.6 torch.compile baseline,
- fused conservative service curve,
- VLM graph replay baseline,
- DiT step CUDA Graph baseline,
- operator inventory,
- admission/horizon simulator configuration.

Exit criteria:

- all reported numbers have exact JSON paths,
- public official timing and local harness timing are not conflated.

### Stage 1: Profiling Gates

Deliverables:

- Nsight Systems traces for official compiled, graph replay, fused conservative,
  and persistent dummy scheduler,
- Nsight Compute top-kernel roofline table,
- CPU launch/sync breakdown,
- gap and boundary accounting.

Exit criteria:

- enough non-math gap is measured to justify MPK beyond CUDA Graph.

### Stage 2: Persistent Scheduler and Request Ring

Deliverables:

- one persistent kernel,
- request descriptor ring,
- completion ring,
- five lanes,
- phase release,
- burst mode,
- credit sweep,
- per-lane active latency measurement.

Exit criteria:

- no per-request host kernel launch,
- phase release produces stable slot interval,
- ring overhead is below the serving deadline budget.

### Stage 3: DiT Real Task Integration

Deliverables:

- DiT single-step task,
- mixed-step task for different denoising step indices,
- four-step loop,
- correctness tests at operator/block/step/action level.

Exit criteria:

- latency close enough to CUDA Graph step that serving benefit is not erased,
- action drift is controlled.

### Stage 4: VLM Postprefill MPK

Deliverables:

- Qwen3 postprefill GQA attention,
- one decoder block,
- postprefill block stack,
- fixed-shape/padded mask semantics,
- RoPE/scale/slice correctness.

Exit criteria:

- VLM postprefill task is correct and competitive against graph replay or
  compiled baseline.

### Stage 5: Five-Phase Real VLA Runtime

Deliverables:

- real VLM/DiT tasks inside five-phase runtime,
- per-stage resource credits,
- CPU admission replay integration,
- simulator-to-runtime trace replay.

Exit criteria:

- higher stable robot count or lower queue wait than group-batch serving under
  the same active-latency assumptions,
- better fairness across 10Hz/20Hz/30Hz robots.

### Stage 6: CPU/Edge Co-Processing

Deliverables:

- CPU preprocessing overlap,
- CPU postprocessing overlap,
- safety/admission control-plane overlap,
- optional edge preprocessing and fallback proxy,
- network jitter sweep.

Exit criteria:

- overlap improves throughput or robustness without adding fine-grained CPU/GPU
  synchronization back into the request path.

### Stage 7: Paper-Quality Evaluation

Deliverables:

- all RQs answered,
- ablations complete,
- correctness complete,
- real rollout or high-fidelity simulation success metrics,
- comparison against strong baseline and vendor path where available.

## 14. Expected Contributions

A strong paper contribution list:

1. A new abstraction for predictable VLA serving:
   - lease/deadline/phase scheduling rather than queue-first batching.
2. A GPU-resident persistent data-plane design:
   - request rings,
   - phase lanes,
   - fixed buffers,
   - stage credits.
3. A VLA-aware Mega Kernel roadmap:
   - MPK/Mirage tasks for VLM/DiT,
   - cross-operator fusion,
   - prefetch/preallocation.
4. A serving policy:
   - horizon-aware,
   - fairness-aware,
   - resource-credit-aware.
5. A full evaluation showing:
   - single-inference overhead reduction potential,
   - serving capacity/fairness gains,
   - limitations of naive MPS/batching/CPU offload,
   - correctness and closed-loop impact.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| full VLM mega-kernel is too hard | start with VLM postprefill, not full VLM |
| hand-MPK tasks are slow | use Mirage/CUTLASS shape-specific Tensor Core tasks |
| concurrent lanes slow each other down | measure slowdown and feed it into equal-latency/sensitivity evaluation |
| credit hurts throughput | tune by stage, show credit sweep, avoid over-throttling |
| CPU/GPU ring polling costs too much | benchmark polling, doorbell, mapped pinned memory, and low-frequency CPU checks |
| request predictability is imperfect | model jitter and use slack windows |
| graph replay already captures most benefit | compare CUDA Graph against MPK; focus paper on serving-shape gains if kernel speedup is modest |
| closed-loop success does not improve | incorporate horizon penalty and real rollout metrics early |

## 16. What Not To Claim Yet

Do not claim:

- full VLM+DiT MPK is already implemented,
- hand-MPK DiT is faster than CUDA Graph,
- all `sync wait` is removable CPU synchronization overhead,
- local `55 ms` official-style timing is NVIDIA's universal official number,
- CPU dense offload is beneficial for current GR00T VLM/DiT,
- compute-bound/memory-bound classification is final without Nsight Compute.

Safe claims:

- VLA serving has predictable phase/deadline structure.
- Equal-latency five-phase serving already improves simulated admission,
  queueing, and fairness.
- VLM graph replay shows large removable data-plane gap while preserving math.
- Official clean profiling shows many kernel/operator boundaries and request-path
  launch/runtime exposure.
- Persistent scheduler MVP supports phase release and stage credit as control
  mechanisms.
- Full high-performance VLM+DiT MPK remains implementation work.

## 17. Paper Narrative

A concise top-conference narrative:

```text
Modern VLA models are served in closed-loop robot systems, but current GPU
serving stacks treat them like ordinary neural requests. This loses a key
property: action-chunk control makes future inference needs predictable. We
introduce a predictable VLA serving abstraction that uses action horizon and
phase windows to create GPU-resident inference leases. We then design a
persistent MPK-style data plane that executes phase-shifted lanes with
preallocated buffers, GPU-side request rings, stage-level credits, and
VLM/DiT task fusion. On GR00T N1.6, profiling shows substantial launch and
boundary overhead, while equal-latency serving simulation shows that the
five-phase data-plane shape improves capacity and fairness even without assuming
faster kernels. This suggests that VLA serving should be optimized as a
predictable real-time GPU data-plane problem, not as generic queue batching.
```

## 18. Immediate Next Steps

1. Produce Nsight Systems traces for:
   - official clean torch.compile E2E,
   - VLM graph replay,
   - DiT step CUDA Graph,
   - persistent five-phase dummy scheduler.
2. Produce Nsight Compute roofline table for:
   - VLM attention,
   - VLM MLP/linear,
   - DiT attention,
   - DiT FFN,
   - norm/copy/index kernels.
3. Extend persistent scheduler MVP to a long-running cyclic request ring:
   - no finite burst-only measurement,
   - fixed phase interval,
   - per-lane active latency and queue latency.
4. Replace the slow hand-MPK DiT task with a shape-specific compiled task.
5. Continue VLM postprefill MPK rather than full VLM mega-kernel.
6. Feed real measured lane slowdown back into Admission/Horizon.
7. Add real rollout or high-fidelity simulator success-rate validation.

## 19. Artifact Index

Main synthesis document:

- `docs/VLA_MEGAKERNEL_ROADMAP_AND_RESULTS_20260520.mdd`

This new paper-idea document:

- `docs/VLA_PREDICTABLE_MPK_TOPCONF_IDEA_ROADMAP_EVAL_20260520.md`

Key result files:

- `results/gr00t_vlm_mpk_dataplane_proxy_20260510.json`
- `results/gr00t_official_mpk_potential_graphproxy_20260510.json`
- `results/gr00t_mpk_synthetic_admission_profile_20260511.json`
- `results/gr00t_phase5_equal_latency_admission_20260511.json`
- `results/gr00t_phase5_cyclic_fixedlane_mpk_mvp_20260511.json`
- `results/gr00t_phase5_persistent_credit_mpk_mvp_20260511.json`
- `results/official_clean_groot_n16_3b_torchcompile_launch_sync_20260520.json`
- `results/official_clean_groot_n16_noncompute_overhead_breakdown_20260520.json`
- `results/official_clean_groot_n16_operator_inventory_20260520.json`
- `results/gr00t_cpu_gpu_coprocessing_profile_iter8_20260520.json`
- `results/gr00t_cpu_gpu_coprocessing_dense_proxy_20260520.json`
- `results/gr00t_cpu_gpu_coprocessing_admission_proxy_20260520.json`

Related implementation/probe scripts:

- `src/gr00t/eval/bench_gr00t_vlm_mpk_dataplane_proxy.py`
- `src/gr00t/eval/bench_gr00t_phase5_equal_latency_admission.py`
- `src/gr00t/eval/bench_gr00t_phase5_cyclic_fixedlane_mpk_mvp.py`
- `src/gr00t/eval/bench_gr00t_phase5_persistent_credit_mpk_mvp.py`
- `src/gr00t/eval/bench_official_clean_groot_n16_launch_sync.py`
- `src/gr00t/eval/bench_official_clean_groot_n16_overhead_breakdown.py`
- `src/gr00t/eval/bench_official_clean_groot_n16_operator_inventory.py`
- `src/gr00t/eval/probe_gr00t_vlm_postprefill_mpk_subgraph.py`
- `src/gr00t/eval/probe_gr00t_mpk_task_compile_dit_attention.py`

## 20. Final Position

The most defensible paper position is:

```text
The main novelty is not only a faster VLA kernel. The novelty is recognizing
that closed-loop VLA serving is predictable enough to justify a new GPU-resident
serving data plane. MPK/Mirage is the mechanism for making that data plane
efficient, while horizon/phase/admission is the control logic that makes it
VLA-aware.
```

The near-term publishable claim should be:

```text
Five-phase persistent VLA serving improves admission, queueing, and fairness
under equal-latency assumptions, and profiling shows enough VLM/DiT data-plane
gap to motivate MPK-style implementation.
```

The long-term implementation claim should be:

```text
A full VLA Mega Kernel can combine phase-aware serving, request-ring
synchronization, fixed-buffer preallocation, stage-credit scheduling, and
VLM/DiT task fusion into a single GPU-resident runtime for predictable robot
inference.
```
