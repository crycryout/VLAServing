# GR00T N1.6 的 Mirage MPK 适配状态与当前结论

本文档记录 2026-04-23 这轮 `Mirage Persistent Kernel` 适配 `GR00T N1.6 mini AlternateVLDiT` 的当前状态。

目标不是现在就宣称已经拿到正式的 kernel-level 加速结果，而是明确回答三个问题：

1. 现在到底已经能跑到哪一步
2. 哪些子路径已经数值闭合
3. 为什么目前还不能把 `test_mode` 的时间直接当成真正的 serving latency

对应数据文件：

- `results/gr00t_mpk_minidit_subgraph_status_20260423.json`

---

## 1. 当前验证对象

当前适配和验证的是一个最小但真实的 `GR00T N1.6` 风格 `DiT core`：

1. `2-layer AlternateVLDiT`
2. 第 0 层是 `cross-attention block`
3. 第 1 层是 `self-attention block`
4. 最后带 `AdaLN-style normout + proj_out_2`

验证配置：

- `q_len = 17`
- `kv_len = 64`
- `hidden_dim = 1536`
- `cross_dim = 2048`
- `num_heads = 32`
- `head_dim = 48`
- `dtype = bf16`

---

## 2. 当前已经补齐的 MPK primitive

为了让这条 GR00T mini-DiT 路径跑起来，这轮 MPK 侧已经补了这些任务：

1. `generic layernorm`
2. `AdaLayerNorm`
3. `generic linear`
4. `bias_add`
5. `elementwise_add`
6. `full_attention`
7. `gelu`

这一步的意义是：

1. MPK 不再只支持原来偏 LLM 的少数路径
2. 现在已经能表达 GR00T DiT core 的主要算子图

---

## 3. 子图 correctness 结果

当前最关键的结果不是整模型一次性全过，而是：

**绝大部分子路径已经在容差内闭合。**

| 子图 | `max_abs` | `mean_abs` | 结论 |
| --- | ---: | ---: | --- |
| `block0` | `0.0078` | `0.00069` | pass |
| `block1_attn` | `0.5811` | `0.1184` | pass |
| `block1_ff` | `0.7168` | `0.1490` | pass |
| `twoblock` | `0.7451` | `0.1481` | pass |
| `final_normout` | `0.6406` | `0.0505` | pass |
| `final_head` | `0.1782` | `0.0359` | pass |
| `full_model` | `1.1680` | `0.2634` | 还差一点 |

这里最重要的判断是：

1. 之前的“大偏差来自 block1”这个结论已经不成立
2. `block1` 的 attention 和 FF 都已经单独闭合
3. `twoblock` 也已经闭合
4. 最后的 `normout` 和 `final_head` 也分别闭合
5. 当前剩下的问题，不是某一个 primitive 明显错误，而是：
   - 各段 bf16 写回误差的累计
   - 导致 full path 的 `max_abs` 还在 `1.17`

换句话说：

**现在的问题已经从“跑不起来/某个子图错很多”，收缩成“整条链路累计误差略高于当前阈值”。**

---

## 4. 这意味着什么

这一步已经能支持两个很重要的结论：

### 4.1 MPK 现在已经能表达 GR00T DiT core

这不是概念设计，而是已经实际跑过：

1. cross-attention block
2. self-attention block
3. FF block
4. final output head

所以从系统实现角度看：

**“Mirage 现在完全没法承载 GR00T N1.6 DiT” 这个判断已经不对了。**

### 4.2 下一步不是再盲目补 task，而是处理累计误差

当前最合理的后续方向是：

1. 减少中间 bf16 写回导致的累计误差
2. 看是否要为少数关键缓冲区引入更高精度的内部表示
3. 或者在比较标准上，区分：
   - `subgraph correctness`
   - `full-path tolerance`

---

## 5. 为什么现在还不能发布 Mirage 的“性能提升结果”

我已经做过一轮最小 perf probe，但当前不能把它当正式结果。

原因很直接：

1. 当前主要可用的是 `PersistentKernel test_mode`
2. `test_mode` 每次调用都会重新发起 persistent kernel/scheduler 的 launch
3. 因此它测到的是：
   - `launch + single-pass execution`
   - 而不是 steady-state serving

这会带来一个误导性的现象：

- PyTorch eager twoblock `p50` 大约只有 `1.09 ms`
- MPK `run_test_mode()` 的 `p50` 却在 `565 ms` 左右

这个数字**不能**解释成“MPK 比 PyTorch 慢 500 倍”。

更准确的解释是：

**当前测到的是 test harness 的固定 launch 成本，不是 kernel steady-state throughput/latency。**

所以现在不能发布“Mirage 对 GR00T N1.6 已经拿到多少倍加速”这种说法。

---

## 6. 当前最准确的结论

可以把现在的状态压缩成三句话：

1. `Mirage MPK` 已经把 `GR00T N1.6 mini-DiT core` 的主要子图跑通了。
2. 大部分子图已经在容差内闭合，full path 只剩少量累计误差没有完全压下去。
3. 当前 `test_mode` 只能做 correctness，不适合直接做正式 latency 结论；下一步必须转向 steady-state request path。

---

## 7. 下一步

真正可行的下一步不是继续在 `test_mode` 上抠毫秒，而是：

1. 找到 Mirage 的 `offline/online` steady-state request path
2. 在那个路径上测：
   - 单次 request latency
   - steady-state throughput
   - launch amortization 后的真实收益
3. 再决定是否值得继续把这条 MPK 路线扩到更完整的 GR00T action head

这才是后面能够写成论文或系统结果的性能实验入口。
