# VLA Serving 中的相位控制与相位移动

## 1. 这份文档讲什么

这份文档只讨论一件事：

- 在 `VLA Serving` 里，什么是相位
- 为什么要做相位控制
- 什么情况下应该移动相位
- `Pi05` 和 `GR00T N1.6` 上，相位控制分别起什么作用

这里不讨论模型压缩，也不讨论旧的非 `25/50` `Pi05 AutoHorizon` 语义。

## 2. 什么是相位

对一个闭环机器人来说，相位可以理解成：

- 它下一次推理请求在时间轴上的位置
- 或者，它距离当前 action chunk 耗尽还有多少 action

如果两个机器人频率相同，但一个总是在 `t=0.00s` 发请求，另一个总是在 `t=0.03s` 发请求，那么：

- 它们频率相同
- 但相位不同

在 `VLA Serving` 里，相位不是无关紧要的细节，因为它直接决定：

- 多个请求是否会撞在一起
- 是否能形成 batch
- 预取和模型状态准备有没有时间完成
- 请求是否会在 action chunk 耗尽前返回

## 3. 为什么要做相位控制

传统 serving 更关注：

- 请求来了以后怎么排队
- 如何减少平均排队时延

但 `VLA Serving` 不是这样。

这里的请求有 3 个关键特征：

1. 下一次请求大致可预测
2. 真正的 deadline 是旧 chunk 耗尽，而不是单纯的 queueing delay
3. 系统有时可以在合法窗口内提前触发下一次推理

因此，系统不仅可以“被动服务请求”，还可以：

- 主动调整未来请求在时间线上的位置
- 让请求更适合系统当前的 resident state、带宽和 batch 结构

这就是相位控制的意义。

## 4. 相位控制到底控制什么

相位控制的对象不是模型结构，也不是 action 内容本身，而是：

- 下一次推理请求何时被触发

更准确地说，它控制的是：

- `request_window` 内的触发时刻

也就是：

- 最早什么时候可以合法重规划
- 最晚什么时候必须完成重规划

因此相位控制消费的是：

- `legal replan window`

它本质上是一种把控制语义转成调度自由度的方法。

## 5. 两种不同的相位操作

### 5.1 相位移动

相位移动指的是：

- 把某个机器人的下一次推理请求向前或向后挪一点

目标通常是：

- 给模型状态准备留时间
- 避开 compute 冲突
- 平衡未来一小段时间线上的资源压力

它更像：

- 单机器人的时间位置调整

### 5.2 相位锁定

相位锁定指的是：

- 把一组相同微调模型的机器人请求主动对齐

目标通常是：

- 让这些请求在同一时刻附近到达
- 形成同模型 batch

它更像：

- 多机器人的 cohort 组织

所以：

- 相位移动更偏“错峰”和“留带宽”
- 相位锁定更偏“合批”和“形成 cohort”

## 6. 相位控制的正确目标

相位控制不是为了“把请求移动得越多越好”，而是为了满足下面几件事：

1. `Safety`
   - 不能让机器人耗尽当前 action chunk

2. `Latency`
   - request-to-result 仍要满足实时约束

3. `Model-state feasibility`
   - resident state、prefetch、apply 要能跟上

4. `Batch opportunity`
   - 在有价值时才为同模型形成 batch

5. `Fairness`
   - 不能因为某些模型更容易合批，就长期牺牲其它机器人

因此相位控制从来不是单独运行的，它必须和：

- residency
- prefetch
- compute placement
- admission

一起设计。

## 7. Pi05 上的相位控制

### 7.1 当前语义

当前 `Pi05` 只保留 `25/50` 设定：

- 所有 `<25` 的 horizon 按 `25` 处理
- 所有 `>25` 的 horizon 按 `50` 处理
- 合法提前重规划窗口是 `[25, 50]`

这意味着：

- 系统可以在 `[25, 50]` 的任意 action 位置提前请求下一次推理

### 7.2 Pi05 上相位控制的作用

在 `Pi05` 上，相位控制的主要用途不是合批，而是：

- 给 residency / prefetch 留时间
- 让共享 shell 的低频模型更容易被提前准备好
- 放宽 vGPU lease 的可行域

也就是说，`Pi05` 更偏向：

- `phase movement for state preparation`

而不是：

- `phase lock for batching`

### 7.3 当前实验结论

在当前 `25/50` 主线结果里：

- fixed-4 `request-to-result p95` 约 `43.21ms`
- `hard miss = 0`
- admission 容量约 `32.67`

同时，实验也说明：

- 这版 `Pi05` 里真正带来收益的是 `[25, 50]` 这个更宽的合法窗口
- 不是更激进的 `phase_shift` 本身

所以对 `Pi05` 来说，当前结论是：

- 相位控制重要
- 但收益主要来自 `window widening`
- 而不是复杂的相位重排算法

## 8. GR00T N1.6 上的相位控制

### 8.1 当前语义

`GR00T N1.6` 保留自己的 chunk-level 动态过程：

- 每次推理输出 `16-action chunk`
- 合法 chunk window 更紧
- slack 明显比 `Pi05` 小

这意味着：

- 盲目 phase shift 更容易伤害控制时序

### 8.2 GR00T 上相位控制的作用

在 `GR00T N1.6` 上，相位控制的主要用途是：

- 把相同微调模型的机器人组织成稳定 cohort
- 让请求尽量同相到达
- 形成 same-model batch

也就是说，`GR00T N1.6` 更偏向：

- `phase lock for batching`

而不是：

- 单机器人意义上的广泛 phase movement

### 8.3 当前实验结论

实验结果非常明确：

在 `8` 机器人场景：

- strict horizon:
  - `mean_batch_size = 1.38`
  - `reply_over = 7`
- phase-lock batch:
  - `mean_batch_size = 2.0`
  - `reply_over = 0`
  - `p95 = 47.61ms`

在 `16` 机器人场景：

- strict horizon:
  - `mean_batch_size = 2.20`
  - `reply_over = 15`
- phase-lock batch:
  - `mean_batch_size = 4.0`
  - `reply_over = 0`
  - `p95 = 58.43ms`

这说明：

- 对 `GR00T N1.6`，相位控制是决定性收益点
- 但这个收益来自 `phase-lock batching`
- 不是来自一般意义上的“随便移动相位”

## 9. 为什么 Pi05 和 GR00T 的相位控制不同

两者都属于 chunked-action VLA，但 workload 特性不一样。

### Pi05

更偏：

- 多模型 resident state 管理
- 共享 shell 的状态准备
- 利用 `[25, 50]` 窗口做 reservation / prefetch

因此相位控制更像：

- 给状态准备腾时间

### GR00T N1.6

更偏：

- 同模型 cohort 形成
- batch size 放大
- 紧 chunk budget 下减少 reply-over

因此相位控制更像：

- 给 batch 组织相位

一句话总结：

- `Pi05` 用相位控制换状态准备空间
- `GR00T N1.6` 用相位控制换 batch 机会

## 10. 相位控制什么时候会失败

相位控制不是总是有益的，典型失败场景有：

1. `合法窗口太窄`
   - 可以移动的空间太小

2. `移动后破坏控制时序`
   - 例如 chunk 本来就很紧，继续平移只会造成 reply-over

3. `没有和 resident state 联动`
   - 请求移了，但模型状态还是没准备好

4. `没有和 admission 联动`
   - 系统接入的机器人太多，即使调相位也无解

5. `把 phase shift 当成通用默认策略`
   - 这在 `GR00T N1.6` 上尤其危险

所以相位控制必须遵守一个原则：

- 只有当 phase movement 能提高整体可行性时，才值得做

## 11. 在 VLA-vGPU 抽象里它属于什么

相位控制主要落在两个维度：

- `T`: future compute timeline
- `W`: legal replan window

如果再往系统里展开，它还会影响：

- `B`: batch affinity
- `F`: fairness

所以相位控制不是一个附属小优化，而是：

- 把控制侧的可预测性转成 GPU 调度自由度的核心桥梁

## 12. 当前最稳的结论

可以直接收敛成下面两句：

- 对 `Pi05`，相位控制是有用的，但主要收益来自 `25/50` 语义下更宽的 `[25, 50]` 合法重规划窗口。
- 对 `GR00T N1.6`，相位控制真正有效的形式是 `same-model phase-lock batching`，而不是泛化的激进相位移动。

因此，当前系统里正确的主线不是：

- “统一使用一种 phase shift 算法”

而是：

- `Pi05`: window-aware phase movement for residency/prefetch
- `GR00T N1.6`: cohort-aware phase lock for same-model batching

## 13. 推荐一起看的文档

如果要把这份文档放回整体系统里理解，建议一起看：

1. `docs/VLA_WORKLOAD_GPU_VGPU_ABSTRACTION.md`
2. `docs/PI05_GPU_VIRTUALIZATION.md`
3. `docs/GR00T_N1D6_GPU_VIRTUALIZATION.md`
4. `docs/UNIFIED_CHUNKED_VLA_SERVING_SYSTEM.md`
