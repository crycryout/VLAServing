# GR00T N1.6 的 Stage 划分与 MicroBatch 设计、实验和结果

本文档整理目前围绕 `GR00T N1.6` 做过的两类优化：

1. `Stage partition`
2. `MicroBatch`

重点回答 4 个问题：

1. `request-level batch` 为什么不适合 `VLA serving`
2. `VLM stage partition` 到底验证到了哪一步
3. `DiT denoising-step microbatch` 为什么是有效方向
4. 这些方法单独使用时，哪些能改善吞吐，哪些能真正满足 `100ms` deadline

---

## 1. 背景与目标

`GR00T N1.6` 的一次端到端推理可以粗略分成两段：

1. `VLM backbone`
2. `DiT / action denoising`

如果沿用传统 DL serving 的思路，最自然的做法是：

1. 把整次 request 攒成 batch
2. 或者把 `VLM` 拆成多个 stage，再用 `MPS` 做并发流水

但 `VLA serving` 的目标不是单纯提高吞吐，而是：

1. 端到端 deadline 要守住
2. queueing 不能把视觉与状态输入拖旧
3. 多机器人、多模型情况下还要控制 admission bias

因此，我们分别验证了三条路线：

1. `whole-request batching`
2. `VLM stage partition + MPS / stream pipeline`
3. `DiT step-level microbatch`

---

## 2. 设计思路

### 2.1 request-level batching

最粗粒度的做法是把整个 `GR00T` request 当作一个 batch 单位。

这个方向的问题是：

1. 粒度太粗
2. 在 `eager-only baseline` 下，一个 request 本身已经接近 `100ms`
3. 一旦开始攒 batch，就几乎必然把端到端 latency 推过 deadline

---

### 2.2 VLM stage partition

这里做过两种划分：

1. `coarse two-stage`
   - `prefill`
   - `llm`
2. `operator-level four-stage`
   - `vision`
   - `projector`
   - `fuse`
   - `llm`

这条路线的目标不是证明“拆分本身有效”，而是验证：

1. `VLM` 是否可以被切成多个有意义的 stage
2. stage 之间是否能通过 `MPS` 或多 stream overlap 获得端到端收益
3. 在单卡上做 stage 并发时，是否会因为资源争用和 handoff 反而放大 queueing

---

### 2.3 DiT step-level microbatch

这里把 batch 粒度从“整个 request”改成“单个 denoising step”。

原因很直接：

1. `GR00T N1.6` 的 `DiT` 一次 request 包含固定数量的 denoising steps
2. request-level batch 太粗
3. step-level batch 更容易在不显著增加单次服务时间的前提下做 same-model 聚合

对应的组合路径是：

1. `VLM` 先单独执行
2. `DiT` 拆成 `4` 个 denoising steps
3. 每个 step 做 same-model microbatch

---

## 3. 实验文件

### 3.1 request batch vs microbatch

- 脚本：`src/gr00t/eval/bench_gr00t_request_batch_vs_microbatch_admission.py`
- 结果：`results/gr00t_request_batch_vs_microbatch_admission_20260419.json`

### 3.2 VLM coarse two-stage pipeline

- 脚本：`src/gr00t/eval/bench_gr00t_vlm_coarse_pipeline_mps_stream.py`
- 结果：`results/gr00t_vlm_coarse_pipeline_mps_stream_20260419.json`

### 3.3 VLM operator-level four-stage pipeline

- 脚本：`src/gr00t/eval/bench_gr00t_vlm_operator_pipeline_mps.py`
- 结果：`results/gr00t_vlm_operator_pipeline_mps_20260419.json`

### 3.4 stage-aware microbatch pipeline

- 脚本：`src/gr00t/eval/bench_gr00t_stage_microbatch_pipeline.py`
- 结果：`results/gr00t_stage_microbatch_pipeline_20260419.json`

### 3.5 deadline / phase-control reference

- 脚本：`src/gr00t/eval/bench_gr00t_vlm_deadline_phase_control.py`
- 结果：`results/gr00t_vlm_deadline_phase_control_20260419.json`

### 3.6 fairness reference

- 脚本：`src/gr00t/eval/bench_gr00t_batch_only_fair_admission.py`
- 结果：`results/gr00t_batch_only_fair_admission_20260412.json`

### 3.7 compile reference

- 脚本：`src/gr00t/eval/bench_n1d6_same_model_batch.py`
- 结果：`results/groot_n1d6_bridge_dit_batch1_compile_20260423.json`

### 3.8 VLM partial-compile reference

- 脚本：`src/gr00t/eval/bench_gr00t_vlm_partial_compile.py`
- 结果：`results/gr00t_vlm_partial_compile_20260423.json`

---

## 4. 核心实验结果

### 4.1 request-level batch 太粗，但 `95.5 ms` 只是 eager-only baseline

来自 `groot_n1d6_bridge_gr1_component_batch_curves_20260412.json` 与 `groot_n1d6_bridge_dit_batch1_compile_20260423.json`：

| 指标 | 数值 |
| --- | --- |
| eager whole-request batch=1 | `95.50 ms` |
| eager whole-request batch=2 | `125.36 ms` |
| eager whole-request batch=4 | `111.31 ms` |
| eager whole-request batch=8 | `147.73 ms` |
| eager `VLM(batch=1)` | `27.62 ms` |
| eager `DiT(batch=1)` | `62.91 ms` |
| full `get_action(batch=1)` with only diffusion compiled | `39.05 ms` |
| pure `DiT(batch=1, denoising_step=4)` compiled | `18.23 ms` |
| component-wise `eager VLM + compiled pure DiT` | `45.85 ms` |

结论：

1. 之前文档里引用的 `95.50 ms` 应被理解为 `eager-only whole-request baseline`，不是 compile 后的最终数字。
2. 之前单独引用的 `39.05 ms` 不是 pure `DiT`，而是完整 `get_action()` 路径上“只编译 diffusion 子模块”的测量。
3. pure `DiT` 在 `denoising_step=4` 下做 `torch.compile` 后，`batch=1 p50` 已经压到 `18.23 ms`。
4. 从 `batch=2` 开始，`eager` 路径下单次服务时间本身就超过 `100ms`。
5. 因此 `whole-request batching` 仍然不适合作为 `GR00T VLA` 的主 serving primitive。

---

### 4.2 DiT step-level microbatch 的基础曲线是好的

来自 `gr00t_stage_microbatch_pipeline_20260419.json` 的组件元数据：

| 指标 | 数值 |
| --- | --- |
| `VLM(batch=1)` | `27.62 ms` |
| `DiT step batch=1` | `16.36 ms` |
| `DiT step batch=4` | `16.14 ms` |
| `DiT step batch=8` | `16.01 ms` |

结论：

1. `DiT` 的单 step latency 随 batch 增长非常平缓
2. 这说明把 batch 粒度改成 step 是合理的
3. `MicroBatch` 真正有效的位置在 `DiT`，而不是整次 request

---

### 4.3 stage-aware microbatch pipeline 明显优于 whole-request batch，但单独还不够

来自 `gr00t_stage_microbatch_pipeline_20260419.json`：

| 场景 | 路径 | `reply_over_chunk_actions_count` | `mean_chunk_elapsed_p95_ms` | `mean_batch_size` | `stable_under_100ms` |
| --- | --- | ---: | ---: | ---: | --- |
| `1x_per_model` | whole-request batch | `6` | `700.00` | `1.00` | `False` |
| `1x_per_model` | stage microbatch | `0` | `654.22` | `1.00` | `False` |
| `2x_per_model` | whole-request batch | `12` | `700.00` | `2.00` | `False` |
| `2x_per_model` | stage microbatch | `6` | `262.54` | `2.00` | `False` |
| `4x_per_model` | whole-request batch | `24` | `700.00` | `4.00` | `False` |
| `4x_per_model` | stage microbatch | `35` | `218.64` | `1.92` | `False` |

结论：

1. `stage microbatch` 明显降低了 `chunk_elapsed_p95`
2. 在 `2x_per_model` 场景里，`700 ms -> 262.54 ms`，改善非常明显
3. 但单独依赖 stage-aware microbatch 仍然无法把系统压进 `100ms deadline`
4. 这说明它是有效的执行优化，但不是完整的 serving 策略

---

### 4.4 保守 admission 估计进一步证明：microbatch 不能替代 phase control

来自 `gr00t_request_batch_vs_microbatch_admission_20260419.json`：

| 策略 | `mean_final_robot_count` | `mean_p95_ms` | `mean_reply_over` | `mean_batch_size` | `stable_group_ratio` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `whole_request_greedy` | `4.0` | `95.50` | `4.0` | `1.0` | `0.0` |
| `microbatch_greedy` | `4.0` | `93.04` | `4.0` | `1.0` | `0.0` |
| `microbatch_quota_fair` | `4.0` | `93.04` | `4.0` | `1.0` | `0.0` |

这份实验的 `policy_note` 已经说明：

- `microbatch_equivalent` 是保守近似
- 用的是 `VLM(batch1) + 4 * DiT-step(torch_superbatch)`

但即便如此，结果仍然说明：

1. 只改执行时间模型，不改请求相位与 cohort 形成机制
2. admission 层面并不会自然得到更高容量
3. `MicroBatch` 不是 `phase control` 的替代品
4. 这里的 `whole_request_greedy mean_p95 = 95.50 ms` 同样应理解为 `eager-only baseline`，不是 compile 后的最终 request latency

---

### 4.5 VLM 的 stage split 在顺序执行下成立，但 `compile` 与 `MPS` 要分开看

#### 4.5.1 operator-level four-stage

来自 `gr00t_vlm_operator_pipeline_mps_20260419.json`：

顺序执行：

| 指标 | 数值 |
| --- | --- |
| monolithic `p50` | `27.70 ms` |
| monolithic `p95` | `31.13 ms` |
| staged total `p50` | `26.69 ms` |
| staged total `p95` | `26.90 ms` |
| vision mean | `12.87 ms` |
| projector mean | `0.14 ms` |
| fuse mean | `0.15 ms` |
| llm mean | `13.32 ms` |
| output diff | `0.0` |

这说明：

1. `VLM` 的四段拆分在语义和数值上是成立的
2. `stage split` 本身没有问题

但 `pipeline + MPS` 的最佳结果是：

| 指标 | 数值 |
| --- | --- |
| 最优 split | `vision/projector/fuse/llm = 25/25/25/25` |
| latency `p50` | `683.19 ms` |
| latency `p95` | `925.25 ms` |
| throughput | `7.64 req/s` |
| `stable_under_100ms` | `False` |

结论：

1. `operator-level MPS pipeline` 在单卡上没有形成可用的低延迟 serving 路径
2. 根因不是“不能拆”，而是“拆完以后同卡并发导致资源争用与 queueing 放大”

#### 4.5.2 partial compile：`vision eager + llm compiled`

来自 `gr00t_vlm_partial_compile_20260423.json`。

这次不再尝试直接 compile 整个 `VLM backbone`，而是只做：

- `vision/projector/fuse` 保持 eager
- `llm body` 单独 `torch.compile`
- 再用 staged runtime 重新测整次 request

结果如下：

| 指标 | 数值 |
| --- | --- |
| eager monolithic `p50` | `33.00 ms` |
| eager staged `p50` | `32.34 ms` |
| partial compile full-request `p50` | `25.64 ms` |
| eager staged `llm p50` | `16.14 ms` |
| partial compile `llm p50` | `11.94 ms` |

从 `p50` 看，当前软件栈下：

- `32.34 ms -> 25.64 ms`
- 约下降 `6.70 ms`
- 相对下降约 `20.7%`

但要注意，当前只验证了“能跑”和“更快”，还没有把数值一致性完全做实：

- hidden-state `mean_abs diff = 0.0508`
- hidden-state `max_abs diff = 71.5`

因此更准确的结论是：

1. `partial compile` 是当前 `VLM` 路径里真正可跑的 compile 方向
2. 它已经能带来正的端到端收益
3. 但还需要补 action-level 或最终输出级别的一致性验证，才能把它当成 production-safe 路径

#### 4.5.3 coarse two-stage

来自 `gr00t_vlm_coarse_pipeline_mps_stream_20260419.json`：

顺序执行：

| 指标 | 数值 |
| --- | --- |
| monolithic `p50` | `28.25 ms` |
| monolithic `p95` | `28.85 ms` |
| coarse two-stage `p50` | `27.76 ms` |
| coarse two-stage `p95` | `34.14 ms` |
| prefill mean | `13.55 ms` |
| llm mean | `15.76 ms` |
| output diff | `0.0` |

这说明：

1. `two-stage split` 本身也成立
2. 拆分本身并不会让单 request 变慢很多

但并发化以后：

| 路径 | `p50` | `p95` | throughput | `stable_under_100ms` |
| --- | ---: | ---: | ---: | --- |
| best two-stage MPS | `739.64 ms` | `794.54 ms` | `9.99 req/s` | `False` |
| same-process two-stream | `132.47 ms` | `224.60 ms` | `34.00 req/s` | `False` |

结论：

1. 真正有问题的不是 `two-stage split`
2. 真正有问题的是“同卡 stage 并发化的方式”
3. `MPS + 多进程 handoff` 最差
4. 单进程双 stream 会好一些，但依然守不住 `100ms`

---

### 4.6 若目标是 `100ms deadline`，仍然需要 phase control

来自 `gr00t_vlm_deadline_phase_control_20260419.json`：

单请求服务时间：

| 指标 | 数值 |
| --- | --- |
| `p50` | `26.64 ms` |
| `p95` | `32.69 ms` |
| `max` | `35.74 ms` |

4 机器人、周期 `500ms`：

| 场景 | `p95 latency` | `deadline_miss_count` | `stable_under_deadline` |
| --- | ---: | ---: | --- |
| `robots4_burst` | `112.48 ms` | `32` | `False` |
| `robots4_even_phase` | `32.89 ms` | `0` | `True` |

更进一步的 equal-phase search 表明：

| 机器人数量 | `p95 latency` | `deadline_miss_count` | 结果 |
| --- | ---: | ---: | --- |
| `18` | `37.63 ms` | `0` | 稳定 |
| `19` | `828.79 ms` | `559` | 失稳 |

结论：

1. `MicroBatch` 和 `stage split` 可以改善执行路径
2. 但要真正满足 `100ms deadline`，还必须显式处理请求相位
3. 当前最可靠的闭环仍然是：`低单次时延 + phase control + admission`

---

### 4.7 batch-first greedy 会引入 admission bias，fairness 不能省

来自 `gr00t_batch_only_fair_admission_20260412.json`：

| 策略 | `mean_final_robot_count` | `mean_p95_ms` | `accept_rate_gap` | `final_count_gap` | `stable_group_ratio` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_greedy` | `22.83` | `90.01` | `0.2921` | `18` | `1.0` |
| `quota_fair` | `15.67` | `58.43` | `0.0994` | `2` | `1.0` |

`baseline_greedy` 的 `accept_rate_by_type`：

- `30Hz::30hz_bridge = 0.7805`
- `10Hz::10hz_rel30k = 0.7358`
- `20Hz::20hz_fractal = 0.4884`
- `10Hz::10hz_libero = 0.4884`

结论：

1. 单纯按 batch opportunity 做 greedy admission，会偏向更容易形成 cohort 的类型
2. `MicroBatch` 可以提升执行效率，但不能替代 `fairness-aware admission`
3. 如果目标是一个长期稳定的多机器人系统，`fairness` 必须是一等目标

---

### 4.8 真实 unified runtime：`VLM multi-stage + compiled DiT step-microbatch`

前面的 `stage-aware microbatch pipeline` 结果，主要是基于组件实测曲线做的系统级仿真。

为了确认这些结论在真实 runtime 里是否仍然成立，这次又补做了一版真正的统一执行路径：

1. `VLM` 按 `vision / projector / fuse / llm` 四段顺序执行
2. `DiT` 按 `4` 个 denoising steps 执行
3. 每个 denoising step 做 same-model microbatch
4. 当前只打开 `DiT torch.compile`
5. `llm body compile` 暂时不并进这条 runtime，避免和数值一致性问题混在一起

对应脚本与结果：

- 脚本：`src/gr00t/eval/bench_gr00t_unified_multistage_microbatch_runtime.py`
- 结果 1：`results/gr00t_unified_multistage_microbatch_runtime_c2_20260423.json`
- 结果 2：`results/gr00t_unified_multistage_microbatch_runtime_20260423.json`

先看 correctness：

| 指标 | 数值 |
| --- | --- |
| `step_microbatch_vs_reference max_abs` | `0.0` |

这说明当前 runtime 至少在 `DiT step-level microbatch` 这一层，和 reference loop 是数值一致的。

#### 4.8.1 小规模真实 replay：`1 cohort x 2 requests`

来自 `gr00t_unified_multistage_microbatch_runtime_c2_20260423.json`：

burst 场景：

| burst size | eager whole-request `p50` | unified `request_to_result p50` | unified `VLM p50` | unified `DiT-step p50` | mean step batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `95.51 ms` | `56.26 ms` | `32.65 ms` | `5.65 ms` | `1.0` |
| `2` | `106.82 ms` | `88.51 ms` | `31.16 ms` | `6.49 ms` | `2.0` |

phase-locked replay：

| 路径 | `request_to_result p50` | `deadline_miss_ratio` |
| --- | ---: | ---: |
| eager whole-request | `97.87 ms` | `0.5` |
| unified runtime | `86.51 ms` | `0.0` |

结论：

1. 在 `2 slots / 1 cohort x 2 requests` 这一规模下，真实 unified runtime 是有效的
2. 它不仅比 eager baseline 更快，而且已经能把 replay 从“有 deadline miss”拉到“无 miss”
3. 此时 `DiT step batch` 的平均大小已经达到 `2.0`

#### 4.8.2 更大真实 replay：`2 cohorts x 2 requests`

来自 `gr00t_unified_multistage_microbatch_runtime_20260423.json`：

burst 场景：

| burst size | eager whole-request `p50` | unified `request_to_result p50` | unified `VLM p50` | unified `DiT-step p50` | mean step batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `120.62 ms` | `64.59 ms` | `41.63 ms` | `5.63 ms` | `1.0` |
| `2` | `174.45 ms` | `106.77 ms` | `40.07 ms` | `6.59 ms` | `2.0` |
| `4` | `118.24 ms` | `154.03 ms` | `38.48 ms` | `6.20 ms` | `2.0` |

phase-locked replay：

| 路径 | `request_to_result p50` | `deadline_miss_ratio` |
| --- | ---: | ---: |
| eager whole-request | `159.67 ms` | `1.0` |
| unified runtime | `434.12 ms` | `1.0` |

结论：

1. 一旦扩到 `4 slots / 2 cohorts x 2 requests`，真实 unified runtime 反而失稳
2. 根因不是 `DiT step microbatch` 无效
3. 真正拖垮系统的是：
   - `VLM staged runtime` 的固定开销
   - step-level 调度与同步开销
   - 当两个 cohort 同时推进时，`VLM` 和 `DiT` 的交错执行并没有形成足够强的 overlap
4. 这和 earlier simulation 的差异很关键：曲线级仿真低估了真实 runtime 开销

因此更准确的结论是：

1. `stage-aware microbatch` 作为执行优化方向是成立的
2. 但它必须以“真实 runtime 的固定开销”来重新评估
3. 当前这版统一实现只在小规模 cohort 下成立，还不能直接外推到更大的闭环 serving 场景

---

## 5. 总结：哪些结论已经被验证

### 5.1 已经被验证成立的结论

1. `whole-request batching` 对 `GR00T VLA` 来说粒度太粗
2. `VLM` 可以被语义正确地拆成多个 stage
3. `DiT` 的正确 batch 粒度是 `denoising step`
4. `stage-aware microbatch` 的确能显著降低 backlog 和 chunk elapsed time
5. `batch-first greedy` 的确会引入明显的 admission bias
6. pure `DiT torch.compile` 会把 `batch=1, denoising_step=4` 从 `63.83 ms` 压到 `18.23 ms`，因此旧文档里的 `95.5 ms` 不应再被当成“当前最优单请求时延”
7. 真实 unified runtime 已经证明：`VLM multi-stage + compiled DiT step-microbatch` 在小规模 `1 cohort x 2 requests` 下可以做到 `86.51 ms p50 / 0 miss`

### 5.2 目前被明确验证为“不行”的路线

1. 单卡上的 `VLM operator-level MPS pipeline`
2. 单卡上的 `VLM coarse two-stage MPS pipeline`
3. 仅靠 `stage split` 或 `microbatch` 而不做 phase control，就想稳定守住 `100ms deadline`
4. 把 curve-driven stage-aware microbatch 仿真结果，直接当成更大规模 unified runtime 的真实上界

### 5.3 当前最准确的系统判断

`GR00T N1.6` 的这套优化不应该被描述成：

- “只要做 stage 划分和 microbatch，就能自然满足 VLA deadline”

而应该被描述成：

1. `VLM stage split` 主要提供结构分析与后续 runtime 设计基础
2. `DiT step-level microbatch` 是已经验证有效的执行优化
3. 真正能把系统带进 `100ms deadline` 的，仍然是：
   - 低单次时延路径
   - phase control
   - admission
   - fairness-aware scheduling

---

## 6. 当前最稳妥的归纳

一句话总结：

**对 `GR00T N1.6` 来说，`stage partition` 是必要的结构化分析工具，`DiT step-level microbatch` 是有效的执行优化，但它们本身还不是完整的 deadline-safe serving runtime；真正的闭环方案仍然需要和 `phase control + admission + fairness` 联合使用。**
