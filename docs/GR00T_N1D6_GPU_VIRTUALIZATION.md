# GR00T N1.6 GPU 虚拟化与 VLA Serving 说明

## 1. 问题定义

这里研究的不是普通的在线排队式 serving，而是 `GR00T N1.6` 的闭环 `VLA Serving`：

- 每个机器人会周期性地产生推理请求
- 每次推理生成一段 `16-action chunk`
- 如果机器人在旧 chunk 用完前拿不到新 chunk，就发生断供
- `GR00T N1.6` 的 chunk slack 比 `Pi05` 更紧，因此调度错误更容易直接变成 reply-over

因此系统关注的不只是算力，还包括：

- GPU 时间维计算资源
- 同模型机器人形成 batch 的机会
- 多微调模型之间的共享前缀常驻
- admission 的公平性

## 2. 模型与机器人绑定

在当前 `GR00T N1.6` GPU 虚拟化实验里，频率和模型固定绑定：

- `30Hz -> 30hz_bridge`
- `20Hz -> 20hz_fractal`
- `10Hz -> 10hz_libero`
- `10Hz -> 10hz_rel30k`

当前测量常数是：

- compiled inference 约 `43.88ms`
- chunk size `16`
- resident model-state footprint：
  - `bridge/fractal` 约 `6.12 GiB`
  - `libero/rel30k` 约 `8.56 GiB`

这说明：

- `GR00T N1.6` 的原始推理时间已经占了 `100ms` budget 的很大一部分
- 如果继续使用不考虑相位和 batch 的 generic 方法，控制侧很容易直接超时

## 3. 当前系统设计

### 3.1 Shared-prefix Residency

当前 `GR00T N1.6` 路径的基础结构是：

- 把多个微调模型的共享前缀状态常驻在 GPU
- 把任务相关状态作为各模型的私有 resident state
- 在此基础上做 same-model batching

这里的 `shared-prefix residency` 指的是：

- 结构共享的 resident state
- 已经在当前实验里验证有效

它不是：

- 早先那些压缩 / 去重 / 无损解码方案

所以这条路线的核心不是“把模型压缩后再恢复”，而是：

- 对 `GR00T N1.6` 可共享的结构部分直接常驻
- 让后续调度主要围绕 batch 和 phase 组织

### 3.2 Same-model Phase-lock Batching

`GR00T N1.6` 的第一性收益点不是任意 phase shift，而是：

- 把相同微调模型的机器人请求主动锁相
- 让这些请求尽量组成稳定 batch

系统不会盲目地把所有请求往前或往后推，而是：

- 优先保留 chunk 安全性
- 只在合法 chunk window 内做相位对齐
- 把 phase control 用来形成同模型 cohort

这条路线的直接收益是：

- batch size 上升
- reply-over 降低
- 在不增加 request-to-result p95 的情况下让更多机器人稳定服务

### 3.3 Quota-fair Admission

只做 batch-first greedy admission 会有明显偏置：

- 更容易组成大 batch 的模型会被优先接纳
- 数量少、难成 batch 的模型会被系统性压制

因此当前 `GR00T N1.6` 路径使用：

- `quota-fair admission`

它的目标不是盲目最大化 admitted robot 数，而是同时控制：

- accept-rate gap
- final-count gap
- p95 latency
- reply-over

也就是说，`GR00T N1.6` 这里的 GPU 虚拟化不是只做吞吐优化，而是：

- 按 cohort、batch opportunity 和 fairness 一起做 lease 分配

### 3.4 MPS 的角色

我们也试过：

- `batch only`
- `batch + MPS`

当前结果表明：

- `MPS` 在这条 `GR00T N1.6` runtime 上没有额外提升稳定容量

所以当前主线不是：

- `MPS partition`

而是：

- `shared-prefix residency + same-model phase-lock batching + fair admission`

## 4. 关键脚本与结果

### Shared-prefix + phase-lock runtime

- `src/gr00t/eval/bench_gr00t_shared_prefix_phase_lock_batch_mps.py`
- `results/gr00t_shared_prefix_phase_lock_batch_mps_20260412.json`

用途：

- 验证 `GR00T N1.6` 的 shared-prefix resident runtime
- 对比 strict horizon 与 same-model phase-lock batch
- 对比 batch only 与 batch + MPS

### Fair admission

- `src/gr00t/eval/bench_gr00t_batch_only_fair_admission.py`
- `results/gr00t_batch_only_fair_admission_20260412.json`

用途：

- 对比 greedy batch-first admission 和 quota-fair admission
- 衡量 accept-rate gap、final-count gap、p95 和稳定性

### 方法总表

- `src/bench_unified_chunked_vla_effectiveness.py`
- `results/unified_chunked_vla_effectiveness_20260412.json`

用途：

- 汇总 `GR00T N1.6` 的方法级有效性
- 验证 phase-lock、fair admission、MPS ablation

### 和旧方法对照

- `src/bench_unified_chunked_vla_vs_baselines.py`
- `results/unified_chunked_vla_vs_baselines_20260412.json`

用途：

- 把 `GR00T N1.6` 和 `GPUlet-like / Clockwork-like / REEF-like / Paella-like / USHER-like / DistServe-like` 做统一口径对照

## 5. 当前结论

### Same-model Phase-lock 的直接效果

在 `8` 机器人场景：

- strict horizon:
  - `mean_batch_size = 1.38`
  - `reply_over = 7`
- phase-lock batch:
  - `mean_batch_size = 2.0`
  - `reply_over = 0`
  - `request-to-result p95 = 47.61ms`

在 `16` 机器人场景：

- strict horizon:
  - `mean_batch_size = 2.20`
  - `reply_over = 15`
- phase-lock batch:
  - `mean_batch_size = 4.0`
  - `reply_over = 0`
  - `request-to-result p95 = 58.43ms`

这说明：

- `phase-lock batching` 是 `GR00T N1.6` 稳定 serving 的决定性因素
- 收益来自 cohort alignment，而不是更激进的一般性 phase shift

### Fair Admission 的效果

相对 greedy batch-first admission：

- `accept-rate gap: 0.2921 -> 0.0994`
- `final-count gap: 18 -> 2`
- `mean p95: 90.01ms -> 58.43ms`

代价是：

- `mean_final_robot_count: 22.83 -> 15.67`

这说明：

- `quota-fair admission` 本质上是在做公平性和容量之间的显式 trade-off
- 但它显著降低了 admission bias，并保持 `stable_group_ratio = 1.0`

### MPS Ablation

当前 `GR00T N1.6` runtime 下：

- `16` 机器人时，`batch only` 和 `batch + MPS` 完全相同
- `24` 机器人时，两者都不稳定，`reply_over = 18`

所以结论是：

- `MPS` 不是当前主线收益来源

### 和传统方法对照

在统一 baseline 对照里：

- 当前最新 `GR00T` 方法：
  - `4` 机器人：`43.88ms p95`
  - `16` 机器人：`58.43ms p95`
  - 都满足 `100ms`

- generic full-resident upper bound：
  - `140.34ms p95`
  - 本身就超过 `100ms`

- 最佳传统 baseline `usher_like`：
  - `520.65ms p95`

- `GPUlet-like temporal / spatial / spatio-temporal`：
  - 都不可行

这说明：

- 对 `GR00T N1.6` 来说，问题不只是 swap
- 即使 full-resident，也仍然是 compute + horizon constrained
- 旧的 GPU 资源划分方法没有利用 VLA 请求的可预测相位和 same-model batch affinity

## 6. 为什么能套进 VLA-vGPU 抽象

在统一抽象里，`GR00T N1.6` 对应：

- `T`: 未来请求的相位布局
- `S`: batch lane，必要时才考虑 optional MPS share
- `M`: shared-prefix resident state + task-specific resident state
- `W`: 自身合法 chunk window
- `B`: same-model batch affinity
- `F`: fairness weight

因此 `GR00T N1.6` 的 GPU 虚拟化核心不是：

- 给每个机器人切一块固定 GPU

而是：

- 给每个机器人一个可预测的时间 lease
- 给每个模型族一个 shared-prefix resident set
- 给同模型 cohort 一个 batch lane
- 给 admission 一个 fairness 约束

## 7. 当前边界

这条 `GR00T N1.6` 路线当前已经验证：

- shared-prefix residency 有效
- phase-lock batching 有效
- fair admission 有效

但也有明确边界：

- generic full-resident 上界仍然超过 `100ms`
- 说明单靠“所有模型都放进去”并不能解决问题
- 需要 workload-aware 的相位组织和 admission 设计

所以当前主线不应该写成：

- “GR00T 只要显存够就能直接 serve”

而应该写成：

- “GR00T 必须利用 shared-prefix、same-model cohort 和 fair admission，才能接近稳定 serving”

## 8. 推荐阅读顺序

如果只想快速理解当前有效版本，建议按下面顺序看：

1. `docs/VLA_WORKLOAD_GPU_VGPU_ABSTRACTION.md`
2. `docs/UNIFIED_CHUNKED_VLA_SERVING_SYSTEM.md`
3. `src/gr00t/eval/bench_gr00t_shared_prefix_phase_lock_batch_mps.py`
4. `results/gr00t_shared_prefix_phase_lock_batch_mps_20260412.json`
5. `results/gr00t_batch_only_fair_admission_20260412.json`
