# VLA Workload 专用的 GPU 虚拟化抽象

## 1. 目标

这里的目标不是把传统 GPU 虚拟化照搬到 VLA Serving 上，而是重新定义：

- 对 `VLA Robot` 而言，什么才是它真正占用的 GPU 资源
- 对 `VLA Server` 而言，应该虚拟化什么，而不只是“算力份额”

核心结论是：

**VLA 的 GPU 虚拟化对象，不是一个静态的 CUDA context，也不是一个固定的空间 partition。**

它应该是一个面向闭环控制的、可预测的、带模型状态管理能力的 `VLA-vGPU lease`。

---

## 2. 为什么传统 GPU 虚拟化抽象不够

传统 serving / vGPU 抽象，通常默认：

- 请求到达是外生的、不可预测的
- GPU 资源主要是：
  - 计算时间
  - 计算空间
  - 显存容量
- 调度目标主要是：
  - 吞吐
  - 平均延迟
  - SLO miss rate

但 VLA workload 不是这样。

VLA 有 4 个额外特征：

1. `Next request predictable`
   - 当前 chunk 发出去后，下一次请求大致何时发生是可预测的。

2. `Deadline is action exhaustion, not queue delay alone`
   - 真正的 deadline 不是单纯“100ms 返回”，而是“在旧 action chunk 耗尽前必须返回新 chunk”。

3. `Model state is part of scheduling`
   - 多机器人绑定不同微调模型时，模型权重驻留、预取、swap、decode/apply 本身就是一等资源。

4. `Horizon is controllable`
   - 像 AutoHorizon 这类机制使得请求相位不是完全固定的，系统可以在合法窗口内主动平移请求。

所以 VLA 里需要虚拟化的不只是 compute。
还要虚拟化：

- `future compute slots`
- `resident model state`
- `H2D / decode bandwidth`
- `legal replan window`

---

## 3. 总体抽象：VLA-vGPU

定义一个机器人在系统中的 GPU 虚拟化实例：

`VLA-vGPU = <T, S, M, X, W, B, F>`

其中：

- `T`: temporal compute lease
  - 在未来时间线上为这个机器人保留的一组可用 compute slots

- `S`: spatial compute share
  - 这个机器人可使用的空间算力份额
  - 可以是独占、MPS share、batch lane、stream lane

- `M`: model-state residency lease
  - 这个机器人绑定模型在 GPU 上当前常驻的状态
  - 可能是：
    - full shell
    - shared prefix
    - delta pages
    - hot pages

- `X`: state transfer / decode budget
  - 为这个机器人未来请求预留的：
    - CPU->GPU 带宽
    - GPU decode/apply 带宽
    - page activation buffer

- `W`: legal request window
  - 这个机器人下一次推理可以被触发的合法 action window
  - 例如：
    - `Pi05`: 在 `[25, 50]` 内可重规划
    - `GR00T`: 在 `[8, 16]` 内可提前重规划

- `B`: batch affinity
  - 这个机器人与哪些其它机器人有 batch 亲和性
  - 例如只允许与相同微调模型组成 batch

- `F`: fairness / admission weight
  - 这个机器人在 admission 和调度中的公平权重

这个对象才是 VLA workload 中真正的“虚拟 GPU”。

---

## 4. VLA-vGPU 虚拟化的资源维度

### 4.1 时间维算力虚拟化

不是简单 time slicing，而是：

- 维护每个机器人未来的下一次、下下次请求时间
- 对未来时间线做 reservation
- 在 reservation 上做：
  - compute placement
  - prefetch placement
  - decode/apply placement

所以这里虚拟化的是：

- `predictive compute timeline`

而不是传统意义上的 reactive queue。

### 4.2 空间维算力虚拟化

包括：

- 独占 stream / shell
- MPS share
- batch lane
- block-level 并行机会

但 VLA 的空间划分不是第一性原理。

是否做空间 partition，要服从：

- deadline
- model-state movement cost
- batching opportunity

### 4.3 显存虚拟化

VLA 中最重要的不是“谁占了多少显存”，而是：

- 哪些模型状态常驻
- 哪些只以压缩 / delta 形式常驻
- 哪些页需要在未来多久内被激活

所以显存要被拆成 3 类：

1. `Resident Shell Memory`
   - 当前可直接运行推理的 active shell

2. `Compressed / Shared State Memory`
   - shared prefix
   - exact delta
   - hot pages

3. `Transient Activation Memory`
   - decode/apply buffer
   - staging buffer
   - in-flight swap buffer

### 4.4 带宽虚拟化

在 VLA 里，CPU-GPU / GPU 内 decode 带宽必须是一等资源。

因为对多微调模型 serving 而言，很多时候瓶颈不是 infer kernel，而是：

- H2D copy
- delta decode
- page apply/revert

因此带宽也要虚拟化成 lease：

- `copy slot`
- `decode slot`
- `apply slot`

### 4.5 控制窗口虚拟化

VLA 特有的一类资源是：

- “我还能在多少 action 之后再请求”
- “我最早什么时候允许提前请求”

这等价于一个可被调度器消费的 slack budget。

所以：

- `legal replan window`
- `phase correction budget`

也必须进入 vGPU 抽象。

---

## 5. 抽象的核心单位：Robot Lease，而不是 Request

传统 serving 的抽象单位是 request。

VLA 里更合理的抽象单位是：

- `Robot Lease`

因为系统不是一次次独立地服务无关请求，而是在维护一个闭环机器人：

- 它绑定一个模型
- 它有长期频率
- 它有 future timeline
- 它有持续的 residency 价值
- 它可能和其它机器人形成长期 batch cohort

因此 admission 的对象应该是：

- `admit a robot lease`

而不是：

- `admit a request`

---

## 6. 控制面抽象

整个系统可以抽象成 5 个控制面组件。

### 6.1 Timeline Predictor

职责：

- 根据频率、chunk size、AutoHorizon、历史相位
- 预测每个机器人未来请求时间线

输出：

- next-use time
- legal request window
- exhaustion deadline

### 6.2 Residency Manager

职责：

- 决定：
  - 哪些模型全量常驻
  - 哪些模型共享 prefix
  - 哪些模型只保留 hot pages / delta pages

它维护的是：

- `resident set`
- `compressed resident set`
- `active shells`

### 6.3 Transfer Scheduler

职责：

- 安排：
  - H2D prefetch
  - decode/apply
  - revert / replacement

它处理的不是单独请求，而是未来时间线上的 state movement。

### 6.4 Compute Scheduler

职责：

- 选择：
  - 单请求执行
  - same-model batching
  - batch + MPS
  - reserved shell execution

它消费的是：

- ready compute slot
- ready model state
- request window slack

### 6.5 Admission Controller

职责：

- 判断再接一个机器人后，系统是否仍满足：
  - action exhaustion safety
  - request latency bound
  - fairness
  - minimum robot score

---

## 7. 数据面抽象

控制面之外，数据面可以统一成 3 个 stream：

1. `compute stream`
   - 执行当前 block / batch 的推理

2. `prefetch stream`
   - 做 H2D / page fetch

3. `decode/apply stream`
   - 在 GPU 内完成 delta decode、page activation、state apply

因此一个标准的数据面流水线是：

- 当前请求在 `compute stream` 上执行
- 同时为未来请求在 `prefetch stream` 上搬运状态
- 再由 `decode/apply stream` 提前把下一模型页准备好

这就是 VLA 里的“时空流水化虚拟化”。

---

## 8. 两类后端如何映射到同一抽象

### 8.1 Pi05 映射

Pi05 更偏向：

- `memory-state virtualization`

对应关系：

- `T`: 为未来 chunk 预留推理时间
- `M`: three-shell + exact-delta / hot-page residency
- `X`: H2D + decode/apply overlap
- `W`: AutoHorizon 给出的下一次合法触发窗口
- `B`: batching 很弱，不是主要优化点

因此 Pi05 的优化重点是：

- `resident shell design`
- `predictive prefetch`
- `bandwidth scheduling`

### 8.2 GR00T N1.6 映射

GR00T 更偏向：

- `temporal cohort virtualization`

对应关系：

- `T`: 未来请求的相位布局
- `S`: batch lane / optional MPS share
- `M`: shared-prefix resident state
- `W`: 8~16 action 之间可提前重规划
- `B`: same-model batch affinity 非常强

因此 GR00T 的优化重点是：

- `phase lock`
- `same-model batching`
- `quota-fair admission`

---

## 9. 统一的资源记账方式

对每个 admitted robot，系统维护一份 `lease descriptor`：

```text
Lease(robot_i) =
{
  model_id,
  frequency_hz,
  chunk_size,
  next_request_time,
  request_window = [t_open, t_close],
  deadline = t_exhaust,
  resident_state_bytes,
  compressed_state_bytes,
  future_copy_bytes,
  future_decode_bytes,
  compute_service_ms,
  batch_affinity_group,
  fairness_weight
}
```

调度器只要维护这些 lease 的总和，就能判断：

- 是否还能再接机器人
- 哪个模型该多常驻
- 哪个模型该提前预取
- 哪些机器人应被相位对齐

---

## 10. Admission 的系统含义

在这个抽象里，admission 不再只是：

- “当前 GPU utilization 够不够”

而是：

- 在未来若干个控制周期内
- 是否还能同时满足：
  - compute feasibility
  - state movement feasibility
  - memory feasibility
  - deadline feasibility
  - fairness feasibility

也就是说，admission 本质是在做：

- `predictive vGPU lease admission`

---

## 11. 这个抽象下的系统目标

一个 VLA 专用 GPU 虚拟化系统，目标不是最大化 batch 吞吐，而是：

1. `Safety`
   - 任何 admitted robot 都不能耗尽 action chunk

2. `Latency`
   - request-to-result 需满足实时约束

3. `Model-state efficiency`
   - 不因多微调模型导致显存和带宽浪费

4. `Predictive utilization`
   - 利用 VLA 的可预测性，把未来资源预先铺好

5. `Fairness`
   - 不因为某些模型更容易组成 batch，就长期偏置 admission

---

## 12. 最终一句话定义

可以把这套抽象收敛成一句话：

**VLA 专用 GPU 虚拟化，不是给每个机器人一个静态算力分区，而是给每个机器人一个“可预测、可移动、带模型状态 lease 的时空联合 vGPU”。**

更具体地说：

**一个 VLA-vGPU = 未来 compute 时间片 + 显存驻留份额 + 状态搬运带宽 + 合法重规划窗口 + batch 亲和性。**

这才是把 GPU 虚拟化思想真正带入 VLA workload 之后，合理的系统抽象边界。
