# GR00T N1.6：Mirage 与官方 `torch.compile` 端到端推理的当前对比状态

本文档只回答一个问题：

**Mirage 现在能不能在同样“compiled”口径下，跑得比 GR00T N1.6 官方原始推理栈更快？**

这里的“同样 compiled 口径”定义为：

1. 官方原始栈使用 NVIDIA 官方 benchmark 里的 `torch.compile` 语义
2. Mirage 侧不再和 eager 对比，而是只讨论 compiled/runtime-compiled 路径

结论先写在前面：

1. 目前**还不能**证明 `Mirage E2E < official compiled E2E`
2. 当前已经拿到稳定结果的是：
   - 官方原始栈 compiled E2E baseline
   - Mirage 的 mini-DiT 子图 steady-state 结果
3. 当前还**没有**拿到有效的 Mirage full E2E compiled 数字
4. 主阻塞点不是 host/scheduler，而是 Mirage 的 `linear` 路径还没有稳定支持 `GR00T` 当前这组形状

---

## 1. 正确对比口径

这轮之后，比较口径统一成：

### 1.1 官方原始推理栈

使用：

- `Isaac-GR00T/scripts/deployment/benchmark_inference.py`

其中 `torch.compile` 的语义是：

1. 仍然使用官方 `Gr00tPolicy`
2. 仍然走官方 `prepare_input -> backbone -> action_head -> get_action` 路径
3. 只对 `action_head.model.forward` 做 `torch.compile`

这和官方 README 的 benchmark 语义一致。

### 1.2 Mirage 路径

当前 Mirage 还没有接进完整官方 `Gr00tPolicy` E2E 栈。

目前 Mirage 实际能稳定测到的，是：

1. `GR00T mini AlternateVLDiT two-block core`
2. `online_notoken / offline` steady-state request path

因此这轮工作的核心，不是硬凑一个不可信的 E2E 数字，而是先把：

1. 官方 compiled E2E baseline 固化出来
2. Mirage 当前能跑到哪里、为什么还不能形成有效 E2E 对比，写清楚

---

## 2. 官方原始栈的 compiled E2E baseline

这轮已经把官方基线单独落成结果文件：

- `results/gr00t_official_compile_e2e_baseline_20260423.json`

数据来源：

- 官方脚本：`Isaac-GR00T/scripts/deployment/benchmark_inference.py`
- 数据集：`demo_data/gr1.PickNPlace`
- `inference_steps = 4`
- `num_iterations = 8`
- `warmup = 3`

这次实测的结果如下。

### 2.1 官方 eager baseline

| 指标 | 数值 |
| --- | ---: |
| E2E `p50` | `209.7 ms` |
| Data Processing `p50` | `4.55 ms` |
| Backbone `p50` | `139.26 ms` |
| Action Head `p50` | `65.7 ms` |

### 2.2 官方 `torch.compile` baseline

| 指标 | 数值 |
| --- | ---: |
| E2E `p50` | `176.2 ms` |
| Data Processing `p50` | `4.55 ms` |
| Backbone `p50` | `139.26 ms` |
| Action Head `p50` | `32.4 ms` |

这组数的系统含义是：

1. 在这次 pinned run 里，官方 compiled E2E baseline 是 `176.2 ms`
2. 官方的 `torch.compile` 主要加速的是 `Action Head`
3. `Backbone` 仍然是整条 E2E 路径里的主耗时部分

---

## 3. Mirage 当前已经验证到哪一步

这轮 Mirage 侧已有两类结果。

### 3.1 稳定有效的 steady-state 子图结果

对应文件：

- `results/gr00t_mpk_steady_state_runtime_20260423.json`
- `results/gr00t_mpk_runtime_breakdown_20260423.json`

对象不是 full E2E，而是：

- `GR00T N1.6 mini AlternateVLDiT two-block core`

结果：

| 路径 | 数值 |
| --- | ---: |
| PyTorch eager twoblock `p50` | `1.40 ms` |
| Mirage `online_notoken` `p50` | `564.12 ms` |
| Mirage `offline` per-request `p50` | `564.63 ~ 565.97 ms` |

runtime breakdown 结果进一步说明：

1. host enqueue 只有 `0.01 ~ 0.12 ms`
2. scheduler 时间只有 `8 ~ 110 us`
3. 约 `93%` 的 worker 时间都耗在 `TASK_LINEAR_GENERIC`

这说明 Mirage 之前真正慢的主因不是 host，不是 scheduler，而是：

- correctness-first 的 `generic linear fallback`

### 3.2 fast-linear 替换实验

这轮还做了一个关键试探：

1. 把 mini-DiT 图里的 `linear_generic_layer`
2. 临时切到 Mirage 现成的 `linear_layer` fast path

单次成功 launch 的观测结果是：

| 路径 | 观测值 |
| --- | ---: |
| Mirage `generic linear` two-block | `~574.8 ms` |
| Mirage `fast linear` two-block | `~42.5 ms` |

这说明：

1. `linear_generic -> linear fast path` 这个方向本身是对的
2. 一旦不用 generic fallback，Mirage 延迟会出现数量级下降

但是，这个结果**还不能当正式比较结果**，因为它还不稳定：

1. repeated launch 会触发 `illegal memory access`
2. 打开 profiling 也会触发 `illegal memory access`
3. 因此当前还不能把它当成可复现、可提交的 steady-state benchmark

---

## 4. 为什么当前还不能做有效的 Mirage E2E 对比

这里最关键的不是“还没写 benchmark”，而是 Mirage 现成 fast path 本身对当前 `GR00T` 形状不成立。

### 4.1 当前测试图的关键形状

当前 Mirage 这条 mini-DiT 路径使用的是：

- `q_len = 17`
- `kv_len = 64`

对应代码在：

- `src/gr00t/eval/bench_gr00t_mpk_steady_state_runtime.py`

### 4.2 Mirage 现成 `linear` kernel 的限制

在：

- `mirage/include/mirage/persistent_kernel/tasks/ampere/linear.cuh`

里明确写着：

1. `TODO: support NUM_ITERS_M > 1`
2. 等价地说，当前实现假定的有效区间更接近 `BATCH_SIZE <= 16`

而 `GR00T` 当前测试图的 `q_len = 17`，已经越过了这个边界。

这解释了为什么：

1. fast-path 单次 launch 有可能成功
2. 但 repeated launch / profiler 打开时会出现 `illegal memory access`

换句话说：

**Mirage 当前并不是“没有 benchmark”，而是“现成 fast path 还没有稳定支持这组 GR00T 形状”。**

### 4.3 Mirage 还没接到 full official E2E

当前 Mirage 只覆盖到了：

1. mini-DiT 两层 core
2. steady-state request path

还没有覆盖到：

1. 官方 `Gr00tPolicy` 的完整输入准备
2. 官方 `backbone`
3. 官方完整 `action_head`
4. 完整 `get_action()` E2E

因此现在还不能给出一个可信的：

- `Mirage compiled full E2E p50`

也就不能和官方的：

- `official compiled E2E p50 = 176.2 ms`

做正式胜负判断。

---

## 5. 当前最准确的结论

这轮工作的正确结论应该写成：

### 5.1 已经确认的事实

1. 官方原始栈在这次 pinned run 里的 compiled E2E baseline 是 `176.2 ms`
2. Mirage 旧路径之所以慢，主因是 `TASK_LINEAR_GENERIC`
3. 把 `linear_generic` 切到 `linear` fast path 后，Mirage 子图延迟会出现数量级下降

### 5.2 目前还不能声称的事情

1. 不能声称 Mirage 已经快过官方 compiled E2E
2. 不能声称 Mirage 已经拿到了稳定的 full E2E compiled benchmark
3. 不能用当前那次 `~42.5 ms` 的单次 fast-path smoke 直接当正式结果

### 5.3 当前真正的工程结论

如果目标是：

- `Mirage E2E < official compiled E2E`

那么下一步必须先做这件事：

1. 修 Mirage `linear` fast path 对 `BATCH_SIZE > 16` 的支持
2. 让 `q_len = 17` 的 GR00T 形状在 repeated launch 下稳定运行
3. 然后再把 Mirage 路径向 full official E2E 接上

在这之前，继续比较“Mirage E2E vs official compiled E2E”没有意义，因为 Mirage 侧的 E2E benchmark 还不成立。

---

## 6. 一句话总结

**当前已经拿到官方 `torch.compile` 端到端基线 `176.2 ms`，也证明了 Mirage 的主瓶颈确实在 `linear_generic`。但 Mirage 现成 `linear` fast path 还不能稳定支持 `GR00T q_len=17` 这组形状，所以 Mirage 端到端 compiled benchmark 目前还不成立；因此现在还不能证明 Mirage 快于官方 compiled E2E。**
