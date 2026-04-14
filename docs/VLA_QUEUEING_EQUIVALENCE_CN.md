# Queueing 不改变 VLA 问题本质

## 1. 结论

在当前这套 `AutoHorizon + 可合法提前重规划 + 非 FIFO 调度` 的设定下：

- `不允许排队，但允许系统在合法窗口内主动提前处理某些机器人`
- `允许排队，但队列中的请求可以被重排，并且在真正开始推理时刷新输入`

这两种系统表达方式在本质上是等价的。

更准确地说，它们求解的是同一个调度问题：

- 每个机器人在一个合法窗口内需要一次 VLA 推理
- 调度器需要决定这次推理在什么时刻真正开始
- 开始前需要准备模型状态
- 开始时需要使用最新的机器人状态输入、视觉输入和其它必需输入
- 返回结果的时间仍然要满足 AutoHorizon 对应的参考值与惩罚约束

因此，`queueing` 在这里不是一个新的 workload，也不是一个新的系统目标。  
它只是把原来已经存在的 `phase shift / reservation / proactive service` 显式写成了一个 `dispatch queue`。

---

## 2. 为什么这两种说法是等价的

### 2.1 原来的说法

原来的系统可以这样理解：

- 机器人当前 chunk 开始后，系统已经知道未来会需要一次新的推理
- 系统不必严格等到某个固定 action 点才处理
- 只要仍在合法窗口内，就可以主动提前处理这个机器人
- 提前处理的目标是：
  - 避开冲突
  - 提前预取模型状态
  - 与同模型机器人组成 batch
  - 提高整体 admit 数量和稳定性

也就是说，系统本来就在做：

- `在一个合法窗口内选择 dispatch time`

### 2.2 加入 queueing 之后的说法

加入 queueing 之后，可以把同一件事换一种表达：

- 机器人在 chunk 开始后，生成一个待服务的请求 token
- 这个 token 可以先进入系统的 waiting set / queue
- 但这个 token 不要求 FIFO 执行
- 调度器可以根据全局状态决定它真正何时出队
- 真正出队、开始推理时，再刷新最新输入

这时系统做的仍然是：

- `在一个合法窗口内选择 dispatch time`

所以二者的核心决策变量完全相同，没有变成另一个问题。

---

## 3. 什么东西没有变

引入 queueing 之后，真正不变的是下面这些核心组件。

### 3.1 Timeline Predictor

它仍然需要预测：

- 下一次请求何时发生
- 合法请求窗口在哪里
- 最晚什么时候必须返回结果

不管你叫它 `future slot reservation`，还是 `queued token scheduling`，都离不开这一步。

### 3.2 Residency / Prefetch Manager

它仍然要决定：

- 哪些模型状态常驻
- 哪些模型状态要提前预取
- 哪些页或哪些 shell 在未来一段时间内要准备好

排队并不会减少这部分难度，因为 Pi05 这类 workload 的核心瓶颈之一本来就是模型状态准备。

### 3.3 Compute Scheduler

它仍然要决定：

- 现在先服务哪个机器人
- 哪些请求可以合并 batch
- 哪些请求应该延后
- 哪些请求应该抢占更早的 slot

换句话说，queueing 只是在实现上把“候选请求集合”显式化了，调度目标没有变化。

### 3.4 Phase Controller

它仍然要决定：

- 哪些机器人应该提前一点处理
- 哪些机器人应该向后平移一点
- 如何把未来的相位布局调成更适合 residency 或 batching

所以 queueing 和 phase control 不是替代关系，而是两种对同一调度自由度的表达方式。

### 3.5 Admission Controller

它仍然要回答：

- 再加入一个机器人后，未来是否还能稳定服务
- 是否会触发 AutoHorizon 惩罚过大
- 是否会出现 reply-over-chunk 或 action exhaustion 风险
- 是否会造成严重 fairness 偏置

这部分也没有变化。

---

## 4. 为什么说“排队请求 + 出队时刷新输入”才是正确语义

如果把 queueing 解释错了，就会把问题改坏。

错误解释是：

- 只有等到请求真正“到达某个固定 action 点”时，系统才第一次知道这个请求存在
- 在这之前不能排队、不能计划、不能提前布局

如果这么做，会直接破坏我们前面所有关于 VLA 可预测性的利用：

- 无法提前预取模型状态
- 无法提前布局 batch
- 无法提前预留 compute slot
- 很多原本稳定的结果会被人为变成不可行

这不符合 VLA 的真实特点。

更合理的语义是：

- 请求的“需要被服务”这件事是可预测的，因此可以提前进入调度器
- 但用于这次推理的最新状态、视觉观测、历史上下文等输入，不应该在入队时冻结
- 而应该在真正 dispatch 的那一刻刷新和物化

也就是说：

- `queue 里存的是待服务资格`
- `dispatch 时拿的是最新输入`

这与真实机器人闭环更加一致。

---

## 5. 等价成立的前提

这个等价关系不是无条件成立的。它依赖以下前提。

### 5.1 队列不能是 FIFO

如果请求必须严格先来后到，那么它就不再等价于 phase shift 调度。

因为 VLA serving 的最优策略本来就不是 FIFO，而是要综合考虑：

- deadline / slack
- 模型状态 locality
- batch 收益
- residency/prefetch 成本
- fairness

所以必须允许 `non-FIFO dispatch`。

### 5.2 队列里的请求必须是可更新的

对同一个机器人来说，系统更关心的是：

- 当前时刻最新、最有效的一次推理需求

而不是：

- 每个历史时刻都保留一条刚性的 pending request

否则就会把机器人闭环的语义错误地离散化成普通在线服务请求。

### 5.3 输入必须在 dispatch 时刷新

如果入队时就把输入冻结，那么排队越久，输入越旧，系统就不再和原来的主动提前处理等价。

正确做法是：

- 入队时只记录调度对象和约束
- 真正开始推理时，再抓取最新输入

### 5.4 惩罚仍然按旧的 AutoHorizon 规则记账

也就是说，结果返回时间仍然要相对于：

- 当前 chunk 的开始时刻
- chunk 剩余 action 的消耗过程
- AutoHorizon 对应的参考值

来计算惩罚和稳定性。

如果把惩罚目标改成普通在线服务里的“请求提交后 100ms 内必须返回”，那就不是原问题了。

---

## 6. 用 VLA-vGPU 抽象来看这个问题

在 `VLA-vGPU` 抽象下，这个等价关系会更清楚。

一个机器人拿到的不是一次普通 request 的排队资格，而是一份带预测语义的 lease：

- 未来可用的 compute 时间片
- 未来可能常驻的模型状态
- 未来可用的带宽预算
- 合法的请求窗口
- 可利用的 batch 亲和性

不管你把 runtime 写成：

- `reservation + phase shift`

还是写成：

- `queue token + dispatch`

本质上都是在调度同一个 lease。

因此 queueing 不是新的 abstraction；  
它只是把原来的 abstraction 显式展开成一个更接近实现层的数据结构。

---

## 7. 对 Pi05 和 GR00T N1.6 分别意味着什么

### 7.1 Pi05

对 Pi05 来说，queueing 不会把主收益点从 residency/prefetch 变成别的东西。

Pi05 的主收益点仍然是：

- frequency-aware residency
- predictive prefetch
- shell 复用
- 带宽与计算重叠

queueing 的作用只是：

- 让“什么时候真正 dispatch 某个机器人”这件事更显式
- 允许调度器在多个 pending robot 之间更自然地重排顺序

所以 Pi05 的算法核心没有变化。

### 7.2 GR00T N1.6

对 GR00T N1.6 来说，queueing 的主价值是更容易表达：

- same-model phase lock
- batch formation
- fair dispatch

但 GR00T 的主收益点仍然是：

- shared-prefix residency
- same-model batching
- phase alignment
- quota-fair admission

也就是说，queueing 不是新的优化点，只是让这些优化点更容易被 runtime 表达出来。

---

## 8. 最终结论

可以把这件事概括成一句话：

**在 VLA serving 中，queueing 并没有把问题从“主动 phase-shift 调度”变成“普通请求排队”。它只是把原本已经存在的可预测调度自由度，改写成了一个显式的、可重排的 dispatch queue。**

因此：

- `workload 本质没有变`
- `系统目标没有变`
- `核心算法没有变`

真正变化的是 runtime 表达方式：

- 从 `proactive service / reservation`
- 变成 `queue token / non-FIFO dispatch / dispatch-time input refresh`

所以如果实现正确，queueing 不是引入一个新问题，而是在不改变问题本质的前提下，把同一个问题写得更清楚。
