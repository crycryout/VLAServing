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
2. 一个 request 本身已经接近 `100ms`
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

---

## 4. 核心实验结果

### 4.1 request-level batch 太粗

来自 `gr00t_request_batch_vs_microbatch_admission_20260419.json` 和组件曲线：

| 指标 | 数值 |
| --- | --- |
| whole-request batch=1 | `95.50 ms` |
| whole-request batch=2 | `125.36 ms` |
| whole-request batch=4 | `111.31 ms` |
| whole-request batch=8 | `147.73 ms` |

结论：

1. 整次 request 在 `batch=1` 时就已经接近 `100ms`
2. 从 `batch=2` 开始，单次服务时间本身就超过 `100ms`
3. 因此 `whole-request batching` 不适合作为 `GR00T VLA` 的主 serving primitive

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

---

### 4.5 VLM 的 stage split 在顺序执行下成立，但 MPS stage pipeline 不成立

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

#### 4.5.2 coarse two-stage

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

## 5. 总结：哪些结论已经被验证

### 5.1 已经被验证成立的结论

1. `whole-request batching` 对 `GR00T VLA` 来说粒度太粗
2. `VLM` 可以被语义正确地拆成多个 stage
3. `DiT` 的正确 batch 粒度是 `denoising step`
4. `stage-aware microbatch` 的确能显著降低 backlog 和 chunk elapsed time
5. `batch-first greedy` 的确会引入明显的 admission bias

### 5.2 目前被明确验证为“不行”的路线

1. 单卡上的 `VLM operator-level MPS pipeline`
2. 单卡上的 `VLM coarse two-stage MPS pipeline`
3. 仅靠 `stage split` 或 `microbatch` 而不做 phase control，就想稳定守住 `100ms deadline`

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
