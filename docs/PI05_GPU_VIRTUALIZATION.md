# Pi0.5 GPU 虚拟化与 25/50 VLA Serving 说明

## 1. 问题定义

这里研究的不是普通的在线排队式 serving，而是 `Pi0.5` 的闭环 `VLA Serving`：

- 每个机器人会周期性地产生推理请求
- 请求时间在很大程度上是可预测的
- 每次推理生成一段 `50-action chunk`
- 如果机器人在旧 chunk 用完前拿不到新 chunk，就发生断供

因此系统关注的资源不是只有算力，还包括：

- GPU 时间维计算资源
- GPU 显存驻留资源
- CPU-GPU 带宽资源
- 未来可预测请求带来的 reservation / prefetch 机会

## 2. 模型与机器人绑定

在当前 `Pi0.5` GPU 虚拟化实验里，频率和模型固定绑定：

- `30Hz -> 30hz_official_ft`
- `20Hz -> 20hz_quantiles`
- `10Hz -> 10hz_a_logits`
- `10Hz -> 10hz_b_autoh`

这个绑定用于模拟“不同机器人运行不同微调模型”的多模型 serving。

## 3. 当前系统设计

### 3.1 Three-shell 基础结构

GPU 上采用 `3-shell` 结构：

- `Shell-A` 常驻 `30Hz` 模型
- `Shell-B` 常驻 `20Hz` 模型
- `Shell-C` 由两个 `10Hz` 模型共享

设计直觉是：

- 高频模型更值得长期常驻
- 低频模型更适合通过预测式预取完成服务

### 3.2 Frequency-aware Residency

系统对低频模型保留可调的常驻预算：

- `r10a`
- `r10b`

这里不再讨论更细的模型表示实现，而只把它抽象成：

- `10Hz` 模型可以有不同的常驻比例
- 常驻比例越高，未来请求到来时需要补齐的模型状态越少

### 3.3 Predictive Prefetch

因为机器人请求是可预测的，系统不会等请求到来后才考虑模型切换，而是：

- admission 后维护未来请求时间线
- 根据 `next-use` 提前为共享 shell 准备下一个模型
- 让模型状态准备尽量隐藏在前一个请求的计算空窗里

### 3.4 25/50 控制语义

当前文档只保留 `Pi0.5` 的 `25/50` 设定：

- 所有 `<25` 的 horizon 统一按 `25` 处理
- 所有 `>25` 的 horizon 统一按 `50` 处理
- 因而系统实际看到的目标集合是 `{25, 50}`
- 合法提前重规划窗口是 `[25, 50]`

这意味着：

- 调度器不再追踪旧的细粒度 horizon 离散值
- 它只需要保证：
  - 尽量在 `25` 或 `50` 附近完成重规划
  - 且绝不允许在 `50-action exhaustion` 前断供

## 4. 关键脚本与结果

### 固定四机器人 + residency/prefetch

- `src/lerobot/eval/bench_pi05_four_model_residency_prefetch_system.py`
- `results/pi05_four_model_residency_prefetch_system_20260406.json`

用途：

- 验证 `30Hz + 20Hz + 10Hz + 10Hz`
- `30/20` 常驻
- 两个 `10Hz` 共用一个 shell
- 基于真实测量常数的系统级仿真

### 25/50 语义 + phase-shift

- `src/bench_pi05_vla_serving_autoh25_50_phase_shift.py`
- `results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json`

用途：

- 只保留 `Pi0.5` 的 `{25, 50}` 控制语义
- 允许在 `[25, 50]` 内提前触发下一次 VLA 推理
- 对比 `strict_horizon` 和 `phase_shift`
- 评估 fixed-4 和 admission 扩容

## 5. 当前结论

### 固定四机器人

在 `25/50 + reservation + residency/prefetch` 设定下：

- `request-to-result p95 ≈ 43.21ms`
- `hard_miss_count = 0`
- `reply_over_chunk_actions_count = 0`
- `fleet_score = 1.0`
- `min_robot_score = 1.0`

这表示：

- 不会耗尽 `50` 个 action
- 固定四机器人场景是稳定的

### 带 admission 的结果

在同样的 `25/50` 设定下：

- `mean_admitted_total = 32.67`
- `mean_fleet_score = 0.9934`
- `mean_min_robot_score = 0.9644`
- `mean_miss_autohorizon_ratio = 0.0420`
- `request-to-result p95 ≈ 43.21ms`

相比旧设定，主要变化是：

- 不是推理本身更快
- 而是合法重规划窗口变成 `[25, 50]` 后，系统更容易安排 reservation 和 prefetch
- 因而 admission 容量明显提升

### 关于 phase shift

在这版 `25/50` 语义下：

- `phase_shift` 没有继续提升 admission 数量
- 真正带来收益的是 `25/50` 控制语义本身
- 也就是 `legal replan window` 变宽

## 6. 当前边界

这版系统仍然是系统级仿真，不是最终完整在线 runtime：

- residency / prefetch / reservation 都已经建模
- measured constants 已经接入
- 但还不是完整的在线 GPU runtime 实现

因此当前结论应理解为：

- 设计和 measured-constant 仿真已经成立
- 当前主线应该围绕 `25/50 + predictive residency/prefetch` 展开

## 7. 推荐阅读顺序

如果只想快速理解当前有效版本，建议按下面顺序看：

1. `docs/VLA_GPU_VIRTUALIZATION_POLICY_20260412.md`
2. `docs/VLA_WORKLOAD_GPU_VGPU_ABSTRACTION.md`
3. `src/bench_pi05_vla_serving_autoh25_50_phase_shift.py`
4. `results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json`
