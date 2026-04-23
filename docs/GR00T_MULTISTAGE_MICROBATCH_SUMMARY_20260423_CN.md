# GR00T N1.6 的 MultiStage 与 MicroBatch：设计、实验与结论摘要

这份文档是当前 `GR00T N1.6` 上 `MultiStage` 与 `MicroBatch` 工作的摘要版。

如果只想快速回答三个问题：

1. 为什么要做 `MultiStage`
2. 为什么要做 `MicroBatch`
3. 这些方法到底有没有用

看这份即可。

对应的详细长文档在：

- `docs/GR00T_STAGE_PARTITION_AND_MICROBATCH_RESULTS_20260423_CN.md`

---

## 1. 问题背景

`GR00T N1.6` 的一次推理大致包含两段：

1. `VLM backbone`
2. `DiT / action denoising`

如果直接沿用普通 DL serving 的思路，最自然的办法是：

1. 把整次 request 当成一个 batch 单位
2. 或把 `VLM` 硬拆成多个 stage，再在同一张卡上并发跑

但 `VLA serving` 不是普通吞吐问题，它有 4 个额外约束：

1. 端到端推理要守住 `100ms deadline`
2. request 不能因为排队把视觉/状态输入拖旧
3. 多机器人请求是闭环 workload，不是独立无关的 Web request
4. 同模型 cohort、相位控制、admission fairness 都会影响系统稳定性

所以这里要回答的不是“怎么把 GPU 跑满”，而是：

- 什么粒度的 batch 才适合 `GR00T`
- `VLM` 是否值得 stage 化
- `MicroBatch` 到底能解决什么，不能解决什么

---

## 2. 两个核心设计

### 2.1 MultiStage

`MultiStage` 的目标是把 `VLM` 从一个黑盒 request 切成更可控的执行段。

当前主要验证了两种划分：

1. `coarse two-stage`
   - `prefill`
   - `llm`
2. `operator-level four-stage`
   - `vision`
   - `projector`
   - `fuse`
   - `llm`

这条线的核心目的不是“为了拆而拆”，而是验证：

1. `VLM` 在语义和数值上能不能被切开
2. 切开之后顺序执行会不会变慢
3. 切开之后如果同卡并发，会不会真的更快

### 2.2 MicroBatch

`MicroBatch` 的目标是把 batch 粒度从“整次 request”降到“单个 denoising step”。

原因很直接：

1. `whole-request batching` 粒度太粗
2. `DiT` 一次 request 内部本来就有固定 denoising steps
3. 同模型机器人更容易在 step 粒度上组成 batch，而不是在整 request 粒度上组成 batch

这里最重要的判断是：

- `GR00T` 里真正适合做 `MicroBatch` 的部分是 `DiT`
- 不是整条 request，更不是整个 `VLM`

---

## 3. 关键实验

### 3.1 whole-request batch 基线

对应结果：

- `results/groot_n1d6_bridge_gr1_component_batch_curves_20260412.json`
- `results/groot_n1d6_bridge_dit_batch1_compile_20260423.json`
- `results/groot_n1d6_bridge_pure_dit_compile_20260423.json`

关键数字：

| 指标 | 数值 |
| --- | --- |
| eager whole-request `batch=1` | `95.50 ms` |
| eager whole-request `batch=2` | `125.36 ms` |
| eager whole-request `batch=4` | `111.31 ms` |
| eager `VLM(batch=1)` | `27.62 ms` |
| eager `DiT(batch=1)` | `62.91 ms` |
| pure compiled `DiT(batch=1, denoising_step=4)` | `18.23 ms` |

结论：

1. `95.50 ms` 只能视为 `eager-only baseline`
2. `whole-request batching` 从 `batch=2` 开始就明显越过 `100ms`
3. 所以整 request 粒度不是 `GR00T VLA` 的正确 batch 粒度

### 3.2 VLM MultiStage：顺序切分是否成立

对应结果：

- `results/gr00t_vlm_coarse_pipeline_mps_stream_20260419.json`
- `results/gr00t_vlm_operator_pipeline_mps_20260419.json`
- `results/gr00t_vlm_partial_compile_20260423.json`

关键数字：

| 场景 | `p50` |
| --- | ---: |
| VLM monolithic | `27.70 ~ 28.25 ms` |
| coarse two-stage 顺序执行 | `27.76 ms` |
| four-stage 顺序执行 | `26.69 ms` |
| staged + partial compile full-request | `25.64 ms` |

结论：

1. `VLM` 在数值上可以被切成 `two-stage` 或 `four-stage`
2. 顺序切分本身不会把单 request 拖慢
3. `vision eager + llm compiled` 这条 `partial compile` 路线目前是有效的

### 3.3 VLM MultiStage：同卡并发是否成立

同样来自：

- `results/gr00t_vlm_coarse_pipeline_mps_stream_20260419.json`
- `results/gr00t_vlm_operator_pipeline_mps_20260419.json`

关键数字：

| 路径 | `p50` | `p95` | 结论 |
| --- | ---: | ---: | --- |
| best two-stage MPS | `739.64 ms` | `794.54 ms` | 不可用 |
| same-process two-stream | `132.47 ms` | `224.60 ms` | 仍不满足 `100ms` |
| four-stage MPS pipeline | `683.19 ms` | `925.25 ms` | 不可用 |

结论：

1. 有问题的不是 `stage split`
2. 有问题的是“同卡上把多个 stage 并发跑”的方式
3. `MPS + 多进程 handoff` 在这里是明显错误方向

### 3.4 DiT step-level MicroBatch 是否成立

对应结果：

- `results/gr00t_stage_microbatch_pipeline_20260419.json`
- `results/gr00t_superbatch_gather_scatter_kernel_20260414_fullcurve.json`

关键数字：

| 指标 | 数值 |
| --- | --- |
| `DiT step batch=1` | `16.36 ms` |
| `DiT step batch=4` | `16.14 ms` |
| `DiT step batch=8` | `16.01 ms` |

结论：

1. `DiT` 的单 step latency 随 batch 增长非常平缓
2. 这说明把 batch 粒度改成 `denoising step` 是对的
3. `MicroBatch` 在 `DiT` 上是真有效，不是概念噱头

### 3.5 stage-aware microbatch 的系统级效果

对应结果：

- `results/gr00t_stage_microbatch_pipeline_20260419.json`
- `results/gr00t_request_batch_vs_microbatch_admission_20260419.json`

关键数字：

| 场景 | 路径 | `mean_chunk_elapsed_p95_ms` | `stable_under_100ms` |
| --- | --- | ---: | --- |
| `2x_per_model` | whole-request batch | `700.00` | `False` |
| `2x_per_model` | stage microbatch | `262.54` | `False` |

结论：

1. `stage-aware microbatch` 显著降低了 backlog 和 chunk elapsed time
2. 但它单独还不足以把系统稳定压进 `100ms`
3. 所以它是执行优化，不是完整的 serving 策略

### 3.6 unified runtime 真机闭环

对应结果：

- `results/gr00t_unified_multistage_microbatch_runtime_c2_20260423.json`
- `results/gr00t_unified_multistage_microbatch_runtime_20260423.json`

当前真实 runtime 做的是：

1. `VLM` 按 `vision / projector / fuse / llm` 顺序执行
2. `DiT` 按 `4` 个 denoising steps 执行
3. 每个 step 做 same-model microbatch
4. 当前只打开 `DiT torch.compile`

小规模结果，`1 cohort x 2 requests`：

| 路径 | `request_to_result p50` | `deadline_miss_ratio` |
| --- | ---: | ---: |
| eager whole-request | `97.87 ms` | `0.5` |
| unified runtime | `86.51 ms` | `0.0` |

更大规模结果，`2 cohorts x 2 requests`：

| 路径 | `request_to_result p50` | `deadline_miss_ratio` |
| --- | ---: | ---: |
| eager whole-request | `159.67 ms` | `1.0` |
| unified runtime | `434.12 ms` | `1.0` |

结论：

1. 真实 unified runtime 在小规模下已经成立
2. 但一旦扩到多 cohort，当前实现会被固定开销拖垮
3. 之前 curve-driven simulation 明显低估了真实 runtime overhead

### 3.7 如果目标是 `100ms deadline`，还缺什么

对应结果：

- `results/gr00t_vlm_deadline_phase_control_20260419.json`
- `results/gr00t_batch_only_fair_admission_20260412.json`

关键数字：

| 场景 | `p95 latency` | `deadline_miss_count` |
| --- | ---: | ---: |
| `4 robots burst @ 0.5s` | `112.48 ms` | `32` |
| `4 robots even-phase @ 0.5s` | `32.89 ms` | `0` |

以及：

| 策略 | `mean_final_robot_count` | `accept_rate_gap` |
| --- | ---: | ---: |
| `baseline_greedy` | `22.83` | `0.2921` |
| `quota_fair` | `15.67` | `0.0994` |

结论：

1. 只做 `MicroBatch` 或 `MultiStage` 不够
2. 真正要守住 `100ms`，还需要 `phase control`
3. 真正要让 admission 长期可用，还需要 `fairness-aware admission`

---

## 4. 当前已经可以明确写下的结论

### 4.1 已经被验证成立

1. `whole-request batching` 不适合 `GR00T VLA`
2. `VLM` 可以被语义正确地 stage 化
3. `DiT` 的正确 batch 粒度是 `denoising step`
4. `DiT step-level microbatch` 是有效执行优化
5. `VLM partial compile` 当前是可跑且有收益的方向
6. `VLM multi-stage + compiled DiT step-microbatch` 在小规模真实 runtime 下已经有效

### 4.2 已经被验证为不行

1. 单卡 `VLM two-stage MPS pipeline`
2. 单卡 `VLM four-stage operator-level MPS pipeline`
3. 只靠 `MicroBatch` 而不做 `phase control`
4. 直接把 curve-driven simulation 当成更大规模 runtime 的真实性能

### 4.3 当前最准确的系统判断

`MultiStage` 和 `MicroBatch` 不应该被描述成：

- “单独就能让 `GR00T` 满足实时 serving”

而应该被描述成：

1. `MultiStage` 主要负责把 `VLM` 结构打开，找到真正可以优化的边界
2. `MicroBatch` 主要负责把 `DiT` 的 batch 粒度从 request 降到 denoising step
3. 它们一起能显著改善执行路径
4. 但要形成真正稳定的 `VLA serving system`，仍然要和：
   - `phase control`
   - `admission`
   - `fairness-aware scheduling`
   联合使用

---

## 5. 一句话总结

**对 `GR00T N1.6` 来说，`MultiStage` 是把 `VLM` 从黑盒拆成可调度结构，`MicroBatch` 是把 `DiT` 从粗粒度 request batch 改成 step-level batch。两者都有效，但它们本身还不是完整的 deadline-safe serving 系统；真正能闭环稳定工作的方案，仍然需要和 `phase control + admission + fairness` 一起设计。**
