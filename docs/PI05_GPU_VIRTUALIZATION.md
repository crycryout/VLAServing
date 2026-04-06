# Pi0.5 GPU 虚拟化与 AutoHorizon Serving 说明

## 1. 问题定义

这里研究的不是普通的在线排队式 serving，而是 **VLA Serving**：

- 每个机器人会周期性地产生推理请求
- 请求时间在很大程度上是可预测的
- 每次推理生成一段 action chunk
- 对 `Pi0.5` 来说，一次推理生成 `50` 个 action
- 如果机器人在旧 chunk 用完前拿不到新 chunk，就发生断供

因此本项目把问题抽象成：

- GPU 时间维计算资源
- GPU 显存驻留资源
- CPU-GPU 带宽资源
- 以及未来可预测请求带来的 reservation / prefetch 机会

## 2. 模型与机器人绑定

在最后这版 PI0.5 GPU 虚拟化实验里，频率和模型是固定绑定的：

- `30Hz -> 30hz_official_ft`
- `20Hz -> 20hz_quantiles`
- `10Hz -> 10hz_a_logits`
- `10Hz -> 10hz_b_autoh`

这个绑定用于模拟“不同机器人运行不同微调模型”的多模型 serving。

## 3. 系统设计

### 3.1 Three-shell 基础结构

GPU 上采用 `3-shell` 结构：

- `Shell-A`：
  - 常驻 `30Hz` 模型
- `Shell-B`：
  - 常驻 `20Hz` 模型
- `Shell-C`：
  - 由两个 `10Hz` 模型共享

直觉是：

- 高频模型更值得全量常驻
- 低频模型更适合通过预取和换入完成服务

### 3.2 Residency

系统允许对低频模型设置部分常驻比例：

- `r10a`
- `r10b`

含义：

- `r10a = 0.2` 表示 `10hz_a_logits` 有 `20%` 的压缩模型页常驻 GPU
- `r10a = 0.0` 表示这个模型不额外常驻，只靠预测式预取

### 3.3 Predictive Prefetch

因为机器人请求是可预测的，系统不会等请求到来后才考虑模型换入，而是：

- admission 后维护未来请求时间线
- 根据 next-use 提前为共享 shell 准备下一个模型
- 让模型搬运尽量隐藏在前一个请求的计算空窗里

### 3.4 AutoHorizon + Reservation

最终版不再使用“固定消耗 10 个 action 后请求”的简化规则，而是引入了 `Pi0.5 p50 AutoHorizon`：

- chunk 的目标切换点从离散集合中采样：
  - `15`
  - `16`
  - `17`
  - `18`
  - `19`
  - `50`
- 每个 chunk 都有：
  - 一个软目标 Horizon
  - 一个硬截止：`50-action exhaustion`

调度器维护未来 compute reservations：

- 目标是尽量在 Horizon 附近完成推理
- 同时绝不允许在 `50` 个 action 用完前还拿不到新 chunk

## 4. 惩罚定义

对每个 chunk，若实际切换位置为 `a`，AutoHorizon 目标为 `H`，则偏差代价与：

- `|a - H|`

相关，并且：

- Horizon 越小，同样偏差惩罚越大

实现里使用指数型成功率：

- 偏差越大，score 越低
- 最终统计：
  - `fleet_score`
  - `min_robot_score`
  - `miss_autohorizon_ratio`

## 5. 关键脚本

### 固定四机器人 + residency/prefetch

- `src/lerobot/eval/bench_pi05_four_model_residency_prefetch_system.py`

用途：

- 验证 `30Hz + 20Hz + 10Hz + 10Hz`
- `30/20` 全常驻
- 两个 `10Hz` 共用一个 shell
- 基于真实测量常数的系统级仿真

结果文件：

- `results/pi05_four_model_residency_prefetch_system_20260406.json`

### 绑定频率/模型的 admission control

- `src/lerobot/eval/bench_pi05_residency_prefetch_admission_bound.py`

用途：

- 在固定基础四机器人的前提下
- 只允许接纳绑定好的四类机器人
- 评估 residency/prefetch 下还能扩容多少

结果文件：

- `results/pi05_residency_prefetch_admission_bound_20260406.json`

### 最终版：AutoHorizon + reservation + residency/prefetch

- `src/lerobot/eval/bench_pi05_autohorizon_reservation_prefetch.py`

用途：

- 引入 `Pi0.5 p50 AutoHorizon`
- 引入 chunk 级 horizon 目标
- 结合 GPU shell residency、共享 shell 预取、future slot reservation
- 同时考察：
  - 断供
  - Horizon 对齐
  - admission 扩容

结果文件：

- `results/pi05_autohorizon_reservation_prefetch_20260406.json`

## 6. 当前结论

### 固定四机器人

在最后这版 `AutoHorizon + reservation + residency/prefetch` 里：

- `hard_miss_count = 0`
- `reply_over_chunk_actions_count = 0`
- `miss_autohorizon_count = 0`
- `fleet_score = 1.0`
- `min_robot_score = 1.0`

这表示：

- 不会耗尽 `50` 个 action
- 并且能打中 AutoHorizon

### 带 admission 的 quick 结果

当前 quick 结果里，每组总接纳大约：

- `29~31` 台

但 admission 扩容后，质量还没完全稳住：

- `mean_fleet_score ≈ 0.9826`
- `mean_min_robot_score ≈ 0.9479`

所以这版可以说明：

- 方向是成立的
- 但 admission 还需要进一步收紧或继续优化

## 7. 当前边界

这版系统仍然是**系统级仿真**，不是最终完整在线 runtime：

- 模型侧的 residency / prefetch / reservation 都已经建模
- 但还没有把整个 GPU 内 page-level decode/apply runtime 完整重放

因此当前结论应理解为：

- “设计和 measured-constant 仿真已成立”
- 而不是“所有 runtime 细节都已经工程落地”

## 8. 推荐阅读顺序

如果只想快速理解最后版本，建议按下面顺序看：

1. `src/lerobot/eval/bench_pi05_four_model_residency_prefetch_system.py`
2. `src/lerobot/eval/bench_pi05_residency_prefetch_admission_bound.py`
3. `src/lerobot/eval/bench_pi05_autohorizon_reservation_prefetch.py`
4. `results/pi05_autohorizon_reservation_prefetch_20260406.json`
