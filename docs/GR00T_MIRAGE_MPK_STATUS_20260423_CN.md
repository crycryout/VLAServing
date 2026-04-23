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

## 5. steady-state request path benchmark：已经做了，但结果目前很差

前一版结论里说过，`test_mode` 不能直接拿来做性能判断。

这次已经补做了真正的非 `test_mode` benchmark，对应脚本和结果是：

- 脚本：`src/gr00t/eval/bench_gr00t_mpk_steady_state_runtime.py`
- 结果：`results/gr00t_mpk_steady_state_runtime_20260423.json`

运行这份 benchmark 时，本地 Mirage Python wrapper 额外做了一个很小的接口修复：

1. 在 `compile()` 之后，把 `.so` 里已有的 `init_request_func` 也绑定到 `PersistentKernel`

这不是新的 runtime 机制，只是把原本已经存在的 C 接口从 Python 层接出来。

这份 benchmark 走的是两条真正的 request path：

1. `online_notoken`
   - 用来测单 request latency
2. `offline(total_num_requests=N, max_num_batched_requests=1)`
   - 用来测 steady-state throughput

这里测的是已经闭合的 `two-block mini-DiT core`，而不是 full model。

### 5.1 PyTorch eager baseline

| 指标 | 数值 |
| --- | --- |
| eager twoblock `p50` | `1.40 ms` |
| eager twoblock `p95` | `1.95 ms` |

### 5.2 `online_notoken` 单请求 latency

保守配置：`num_workers=1, num_local_schedulers=1`

| 指标 | 数值 |
| --- | --- |
| `p50` | `564.12 ms` |
| `p95` | `578.88 ms` |
| first-run correctness `max_abs` | `1.8867` |

自动配置：`96 workers / 128 schedulers`

| 指标 | 数值 |
| --- | --- |
| `p50` | `565.03 ms` |
| `p95` | `577.84 ms` |
| first-run correctness `max_abs` | `2.0313` |

结论：

1. 自动配置并没有明显优于保守配置
2. 当前问题不是“worker/scheduler 参数还没调好”
3. steady-state path 的数值误差也比 `test_mode` 更大，说明 runtime path 本身也在引入额外问题

### 5.3 `offline` steady-state throughput

`total_num_requests=8`

| 指标 | 数值 |
| --- | --- |
| total elapsed | `4517.05 ms` |
| per-request `p50` | `564.63 ms` |
| throughput | `1.77 req/s` |

`total_num_requests=64`

| 指标 | 数值 |
| --- | --- |
| total elapsed | `36221.80 ms` |
| per-request `p50` | `565.97 ms` |
| throughput | `1.77 req/s` |

这里最关键的观察是：

1. `8 requests` 和 `64 requests` 的 `per-request latency` 几乎一样
2. `throughput` 也几乎不变
3. 说明当前 steady-state path **没有把 launch/scheduler 成本 amortize 掉**

换句话说：

**offline request loop 已经能跑，但它并没有形成有效的 steady-state 加速。**

### 5.4 runtime breakdown profiling：固定开销到底在哪里

在 steady-state benchmark 之后，又补做了一轮 runtime breakdown profiling：

- 脚本：`src/gr00t/eval/profile_gr00t_mpk_runtime_breakdown.py`
- 结果：`results/gr00t_mpk_runtime_breakdown_20260423.json`
- Perfetto trace：
  - `results/gr00t_mpk_online_notoken_conservative_20260423.perfetto-trace`
  - `results/gr00t_mpk_online_notoken_auto_20260423.perfetto-trace`
  - `results/gr00t_mpk_offline_r8_conservative_20260423.perfetto-trace`

这轮 profiling 的目标只有一个：

**把之前“是不是 host launch / scheduler 本身太重”这个问题直接拆开。**

核心结果如下。

`online_notoken` 保守配置：

| 指标 | 数值 |
| --- | --- |
| `init_request_func_wall_ms` | `0.0109 ms` |
| `launch_enqueue_wall_ms` | `0.0674 ms` |
| `gpu_elapsed_ms` | `564.54 ms` |
| `scheduler_total_us` | `8.19 us` |
| `worker_total_us` | `564373.50 us` |
| `TASK_LINEAR_GENERIC` 占 worker 时间 | `93.32%` |

`online_notoken` 自动配置：

| 指标 | 数值 |
| --- | --- |
| `init_request_func_wall_ms` | `0.0111 ms` |
| `launch_enqueue_wall_ms` | `0.1223 ms` |
| `gpu_elapsed_ms` | `565.46 ms` |
| `scheduler_total_us` | `16.38 us` |
| `worker_total_us` | `565175.30 us` |
| `TASK_LINEAR_GENERIC` 占 worker 时间 | `93.42%` |

`offline_r8` 保守配置：

| 指标 | 数值 |
| --- | --- |
| `init_request_func_wall_ms` | `0.0168 ms` |
| `launch_enqueue_wall_ms` | `0.0844 ms` |
| `gpu_elapsed_ms` | `4512.02 ms` |
| `scheduler_total_us` | `109.57 us` |
| `worker_total_us` | `4431478.78 us` |
| `TASK_LINEAR_GENERIC` 占 worker 时间 | `93.21%` |

这轮 profiling 把结论收窄成了很明确的三点：

1. `init_request_func` 和 host enqueue 的 wall time 都是 `0.01~0.12 ms` 量级，几乎可以忽略。
2. scheduler 自身时间只有 `8~110 us`，占总时间不到 `0.003%`，不是主瓶颈。
3. 绝大部分时间都烧在 GPU worker 侧，而其中约 `93%` 又都集中在 `TASK_LINEAR_GENERIC`。

所以，这一轮之后必须把之前的粗结论改掉：

- 之前的粗结论：`Mirage runtime/scheduler 结构本身是主瓶颈`
- 现在的精确结论：**当前主瓶颈是 correctness-first 的 `generic linear` fallback kernel，而不是 host launch 或 scheduler**

也就是说：

1. steady-state request path 确实没有形成有效 amortization
2. 但“没有 amortize 掉”的主要原因，不是 Python/host/scheduler 太慢
3. 而是当前图里的 GEMM 路径仍然落在一个极慢的 `TASK_LINEAR_GENERIC` 实现上

### 5.5 当前的正确结论

所以这条 Mirage 线现在不能写成：

- “steady-state 路径已经能显著加速 GR00T DiT”

而应该写成：

- “steady-state request path 已经打通，但当前 runtime 对这种小而规则的 VLA DiT 图仍然有巨大的固定开销”

这也解释了为什么：

1. `test_mode` 看起来慢
2. 换成真正的 `online/offline` request path 以后，还是慢
3. 并且 `offline` 的多 request 处理也没有显著 amortization

因此现在性能瓶颈的主因已经很明确：

**不是 graph 没编出来，也不是 host/scheduler 太重，而是图里的 `TASK_LINEAR_GENERIC` fallback 过慢。**

---

## 6. 当前最准确的结论

可以把现在的状态压缩成三句话：

1. `Mirage MPK` 已经把 `GR00T N1.6 mini-DiT core` 的主要子图跑通了。
2. 大部分子图已经在容差内闭合，full path 只剩少量累计误差没有完全压下去。
3. steady-state request path 现在也已经测过，但结果表明当前实现对这类 GR00T 小图仍然有约 `565 ms/request` 的巨大成本，而且主耗时集中在 GPU 侧 `TASK_LINEAR_GENERIC`，没有形成有效吞吐增益。

---

## 7. 下一步

真正可行的下一步不是继续在 `test_mode` 上抠毫秒，而是直接针对 `linear` 路径动刀：

1. 先替换 `TASK_LINEAR_GENERIC`，给当前 GR00T 维度接一个真正可用的优化 GEMM path
2. 再重新测 `online/offline`，验证 steady-state path 是否开始出现真实 amortization
3. 如果替换掉 `linear` 以后仍然慢，再进一步检查：
   - 是否需要更 coarse-grained 的 megakernel
   - 是否需要把多个 denoising steps 进一步合并
   - 是否需要完全绕开当前 MPK request runtime，直接做 VLA 专用 persistent execution

这才是后面能够写成论文或系统结果的性能实验入口。
