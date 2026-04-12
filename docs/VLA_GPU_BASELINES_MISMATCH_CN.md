# 为什么主流 GPU 资源划分与虚拟化方法不能很好支持 VLA Workload

## 1. 问题不是“GPU 虚拟化”本身，而是 workload 变了

主流 GPU serving / 资源划分方法，通常默认：

- 请求到达是独立的
- 系统目标主要是吞吐、平均时延和 SLO miss rate
- 请求进来后先排队，再决定 batch 和执行
- 模型权重状态不是调度的一等资源

但 `VLA Workload` 不满足这些假设。

`VLA` 的关键特征是：

1. `下一次请求可预测`
   - 当前 chunk 发出去后，下一次请求大致什么时候来是可预测的

2. `真正的 deadline 是 action chunk 耗尽`
   - 不是简单的“100ms 返回”而已

3. `模型状态本身是资源`
   - 多个微调模型意味着 resident state、prefetch、状态切换都必须纳入调度

4. `控制窗口可被消费`
   - 可以在合法窗口内主动提前或对齐请求相位

因此，VLA 需要的不是 generic request scheduler，而是：

- `predictive timeline + model-state + control-window aware` 的系统

## 2. 主流方法的共同错位点

### 2.1 只看请求，不看机器人闭环

很多方法调度的是：

- 一个个独立请求

但 `VLA` 真正需要调度的是：

- 一个长期存在的 robot lease

因为一个机器人不仅有“当前请求”，还有：

- 未来请求时间线
- 绑定模型
- resident state 价值
- 可提前重规划的窗口

### 2.2 只看 compute，不看 model state

传统方法常把瓶颈理解成：

- kernel compute
- batch throughput
- queueing

但在多微调模型 `VLA Serving` 里，很多时候决定能否按时返回的是：

- 模型状态是否已经在 GPU
- CPU-GPU / GPU 内状态准备是否来得及

### 2.3 时间切片过粗，无法利用控制语义

传统 temporal partition 往往只会：

- 给请求排时间片

但 `VLA` 里真正可利用的是：

- 合法重规划窗口
- 同模型 cohort 的相位对齐机会

这不是普通 time slicing 能表达的。

### 2.4 不理解“不能排队”的真实含义

很多 serving 方法允许：

- 请求进入队列
- 只要平均意义上满足 SLO 即可

但 `VLA` 不一样。

对机器人来说：

- 一旦排队过久，视觉和状态信息就可能过期
- 或者直接耗尽当前 action chunk

所以这里的约束不是“尽量少排队”，而是：

- admitted robots 必须长期稳定不断供

## 3. GPUlet-like 为什么不适合

### 3.1 方法假设的问题

`GPUlet-like` 的核心是：

- 按分区做空间切分
- 在 duty-cycle / temporal cycle 上做 coarse scheduling
- 使用离线调度器决定资源布局

这个思路对 `VLA` 不合适，因为它没有把：

- 模型状态切换
- legal replan window
- 未来相位

当成核心资源。

### 3.2 直接实验结果

在统一对照里：

- `Pi05` 上，`GPUlet-like temporal / spatial / spatio-temporal` 全都不可行
- `GR00T N1.6` 上，三种也都不可行

对 `Pi05`：

- spatial 不可行的直接原因是：
  - per-model partition 或 shell memory 超过单卡预算
  - 代表性 resident shell 约 `7.485 GiB`

对 `GR00T N1.6`：

- spatial 同样不可行：
  - per-model partition 或 shell memory 超出预算
  - 代表性 resident shell 约 `6.12 GiB`

两边的 temporal / spatio-temporal 不可行原因都类似：

- 一旦把模型状态切换的成本算进去
- cycle / duty-window 就很容易超过 `100ms`

### 3.3 公共代码复现的直接证据

我们还用公开 artifact 做了验证：

- `sanity_single_vgg16` 可以正常出结果
- 但一旦换成 VLA 风格输入：
  - `gr00t_two_model_gpulet`
  - `gr00t_four_model_gpulet`
  - `pi05_four_model_scaledx10_gpulet`
  - 全部在 `30s` timeout 内无法给出可用 schedule

另外，`Pi05` 的请求率还有一个额外问题：

- `glet` 的离线调度器只接受整数 request rate
- 但 `Pi05` 在真实 `AutoHorizon` 下，低频机器人均值请求率是小于 `1 RPS` 的

这说明：

- 不是实现细节没调好
- 而是它的输入抽象本身就和 `VLA` 不匹配

## 4. Clockwork-like 为什么不适合

### 4.1 它解决的是“可预测 compute reservation”

`Clockwork-like` 的优点是：

- 能为可预测请求预留 compute 时间

但它没有解决：

- 多微调模型的 resident state
- 状态准备带宽
- 合法重规划窗口的主动消费

### 4.2 实验结果

对 `Pi05`：

- `clockwork_like p95 = 364.8403 ms`
- `hard miss = 31.0`
- `sla miss = 49.0`

对 `GR00T N1.6`：

- `clockwork_like p95 = 437.0074 ms`
- `hard miss = 2.6667`
- `sla miss = 626.3333`

这说明：

- “只预留 compute” 远远不够
- 一旦模型状态和控制 deadline 被纳入系统，Clockwork 的抽象边界就太窄了

## 5. REEF-like 为什么不适合

### 5.1 它关注的是细粒度时间复用

`REEF-like` 主要擅长：

- 更细粒度的时间切换
- 更灵活的 temporal sharing

但 `VLA` 的问题不是单纯时间复用不够细，而是：

- 模型状态没准备好
- 控制 deadline 不允许长队列
- 同模型 cohort 没被利用

### 5.2 实验结果

对 `Pi05`：

- `reef_like_temporal p95 = 752.6661 ms`
- `hard miss = 64.3333`

对 `GR00T N1.6`：

- `reef_like_temporal p95 = 1195.1513 ms`
- `hard miss = 150.6667`

所以：

- 细粒度 preemption 不能替代 model-state aware serving

## 6. Paella-like 为什么不适合

### 6.1 它更像“为吞吐优化的 batch serving”

`Paella-like` 更适合：

- 提高吞吐
- 对 batch 更友好的离线 / 在线调度

但 `VLA` 需要的是：

- 每个机器人自己的 deadline 安全
- 不能因为 batch 机会而牺牲少数机器人

### 6.2 实验结果

在我们的对照里：

- `Pi05 paella_like p95 = 364.8403 ms`
- `GR00T N1.6 paella_like p95 = 437.0074 ms`

这和 `Clockwork-like` 一样，都明显高于 `100ms`。

因此：

- `Paella-like` 的优化方向和 `VLA` 的目标函数不一致

## 7. USHER-like 为什么不适合

### 7.1 它更偏固定空间配额

`USHER-like` 的核心思路更接近：

- 给不同请求或模型一个相对固定的空间份额

但 `VLA` 里真正重要的是：

- 未来请求什么时候来
- 模型状态该提前给谁准备
- 哪些机器人该形成 cohort

固定空间份额不会自动解决这些问题。

### 7.2 实验结果

对 `Pi05`：

- `usher_like p95 = 471.7146 ms`
- `hard miss = 77.6667`

对 `GR00T N1.6`：

- `usher_like p95 = 520.6532 ms`
- `hard miss = 2.0`
- `sla miss = 1015.0`

这说明：

- 固定空间切分无法替代 workload-aware 的时间/状态协同调度

## 8. DistServe-like 为什么不适合

### 8.1 它通过拆分服务链路来换可扩展性

`DistServe-like` 的思路是：

- 把不同阶段解耦
- 通过跨阶段分离来提高吞吐或扩展性

但 `VLA` 的痛点恰恰是：

- critical path 太紧
- 控制 deadline 很短

如果再人为增加阶段边界，只会拉长关键路径。

### 8.2 实验结果

对 `Pi05`：

- `distserve_like p95 = 1980.0776 ms`
- `hard miss = 62.0`

对 `GR00T N1.6`：

- `distserve_like p95 = 2689.9232 ms`
- `hard miss = 134.3333`

这几乎直接说明：

- `DistServe-like` 的系统分解方向和 `VLA` 的 deadline 结构相冲突

## 9. 即使 full-resident 上界也不够，说明问题不是“只差一点优化”

我们还测了 generic full-resident upper bound。

对 `Pi05`：

- `oracle_full_resident p95 = 104.9249 ms`

对 `GR00T N1.6`：

- `oracle_full_resident p95 = 140.3380 ms`

这两个值都高于 `100ms`。

含义非常关键：

- 问题不是“只要把模型全塞进显存就好了”
- 也不是“再把某个旧方法调一下参数就行”

真正的问题是：

- workload 的抽象边界变了

## 10. 为什么我们的 VLA-aware 方法更有效

### 10.1 Pi05

当前主线结果是：

- `service_e2e_p95 = 43.2133 ms`
- fixed-4 `hard miss = 0`
- 在 `{25,50}` + `[25,50]` 窗口下，admission `mean_admitted_total = 32.6667`

也就是说：

- `Pi05` 的收益来自更宽的控制窗口 + predictive residency / prefetch

### 10.2 GR00T N1.6

当前主线结果是：

- `4` 机器人稳定：`43.8806 ms p95`
- `16` 机器人稳定：`58.4292 ms p95`
- `reply_over = 0`

并且：

- `phase-lock batching` 把 `8` 机器人场景的 `reply_over` 从 `7` 降到 `0`
- 把 `16` 机器人场景的 `reply_over` 从 `15` 降到 `0`
- `quota-fair admission` 把 `accept-rate gap` 从 `0.2921` 降到 `0.0994`

也就是说：

- `GR00T N1.6` 的收益来自 `shared-prefix + phase-lock batching + fair admission`

## 11. 最终一句话

主流 GPU 资源划分和虚拟化方法之所以不能很好支持 `VLA Workload`，不是因为它们“不够快”，而是因为它们优化的是错误的对象：

- 它们优化的是独立请求的吞吐和时延
- 而 `VLA` 需要优化的是可预测 robot lease 的未来时间线、模型状态、带宽和合法重规划窗口

这就是为什么 `VLA-aware GPU virtualization` 必须是一个新的抽象，而不是把旧 serving 方法直接套过来。

## 12. 对应结果文件

- `results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json`
- `results/unified_chunked_vla_vs_baselines_20260412.json`
- `results/unified_chunked_vla_effectiveness_20260412.json`
- `results/public_gpu_serving_artifacts_20260411.json`
