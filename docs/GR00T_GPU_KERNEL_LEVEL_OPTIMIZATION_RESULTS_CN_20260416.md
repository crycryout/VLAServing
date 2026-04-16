# GR00T N1.6 GPU Kernel Level Optimization 当前结果总结

本文档只总结截至 2026-04-16 已经实际跑出的、和 GPU kernel / execution path 优化直接相关的结果。  
重点不是通用 `torch.compile`，而是面向 VLA workload 的执行原语设计：

- phase-locked same-model cohort
- prepared execution boundary
- same-model super-batch execution

对应代码和结果都在当前仓库与本机 `Isaac-GR00T` 代码树中可复现。

---

## 1. 当前最强的 kernel-level 主线是什么

目前最有价值的方向不是单独写一个更快的 gather kernel，也不是单独依赖 `torch.compile`。

当前最强主线是：

**phase-locked same-model super-batch execution primitive**

它的核心思想是：

1. 利用 VLA 的相位可控性，把同模型请求对齐成 cohort
2. 在执行边界上，不再逐请求单独跑 action head
3. 改成一次 same-model super-batch 执行
4. 直接减少 repeated launches、per-request CUDA/CPU 调度开销和 queue wait

这条主线已经有 4 类证据支持：

- microbench
- fixed-scheduler replay
- runtime ablation
- profiler

---

## 2. Microbench：收益主要来自 super-batch execution，不是 pack kernel 本身

代码：

- `src/gr00t/eval/bench_gr00t_superbatch_gather_scatter_kernel.py`

结果：

- `results/_tmp_gr00t_superbatch_gather_scatter_kernel.json`
- `results/gr00t_superbatch_gather_scatter_kernel_20260414_fullcurve.json`

关键数字，`batch=4`：

| mode | p50 延迟 |
| --- | --- |
| sequential_single | 70.253 ms |
| torch_superbatch | 16.710 ms |
| triton_superbatch | 16.161 ms |
| torch_pack_only | 0.069 ms |
| triton_pack_only | 0.179 ms |

结论：

1. 真正起主要作用的不是 gather/scatter 这一步更快
2. pack 自身只占极小一部分时间
3. 主收益来自把多个 request 合并成一次 same-model super-batch 执行

---

## 3. Fixed-scheduler replay：execution primitive 直接决定 deadline 能否守住

代码：

- `src/gr00t/eval/bench_gr00t_phase_locked_superbatch_runtime_replay.py`

结果：

- `results/_tmp_gr00t_phase_locked_superbatch_runtime_replay.json`
- `results/_tmp_gr00t_phase_locked_superbatch_runtime_replay_p32.json`
- `results/_tmp_gr00t_phase_locked_superbatch_runtime_replay_p24.json`

### 3.1 基准压力点：40ms cohort period / 20ms phase stride / 100ms deadline

| mode | service p50 | request-to-result p50 | queue p50 | miss ratio |
| --- | --- | --- | --- | --- |
| sequential_single | 68.093 ms | 882.196 ms | 817.954 ms | 0.9688 |
| torch_superbatch | 15.803 ms | 15.859 ms | 0.0 ms | 0.0 |
| triton_superbatch | 16.063 ms | 16.091 ms | 0.0 ms | 0.0 |

### 3.2 更紧压力点：32ms / 16ms

| mode | latency p50 | latency p95 | miss ratio |
| --- | --- | --- | --- |
| sequential_single | 682.708 ms | 1249.761 ms | 0.9583 |
| torch_superbatch | 22.131 ms | 27.897 ms | 0.0 |
| triton_superbatch | 34.849 ms | 41.253 ms | 0.0 |

### 3.3 更紧压力点：24ms / 12ms

| mode | latency p50 | latency p95 | miss ratio |
| --- | --- | --- | --- |
| sequential_single | 664.345 ms | 1229.029 ms | 0.9583 |
| torch_superbatch | 61.767 ms | 101.005 ms | 0.0833 |
| triton_superbatch | 64.354 ms | 108.914 ms | 0.1667 |

结论：

1. 在固定相位的 same-model cohort 下，execution primitive 本身就足以改变系统是否崩溃
2. 这已经不是“小幅 kernel 加速”，而是直接改变 `deadline miss` 和 `queue collapse`
3. 当前最稳的实现还是 `torch_superbatch`
4. Triton 版本可作为补充实现，但不是当前最强主证据

---

## 4. Runtime ablation：super-batch primitive 可以把稳定 admission 上限翻倍

代码：

- `src/gr00t/eval/bench_gr00t_phase_lock_superbatch_runtime_ablation.py`

结果：

- `results/gr00t_phase_lock_superbatch_runtime_ablation_20260414.json`
- `results/_tmp_gr00t_phase_lock_superbatch_runtime_ablation_batch_plus_mps.json`

### 4.1 batch_only

| mode | 最大稳定机器人数量 | 最佳场景 | p95 |
| --- | --- | --- | --- |
| sequential_single | 16 | 4x_per_model | 64.252 ms |
| torch_superbatch | 32 | 8x_per_model | 16.014 ms |
| triton_superbatch | 32 | 8x_per_model | 16.326 ms |

### 4.2 batch_plus_mps

结果和 `batch_only` 基本一致：

| mode | 最大稳定机器人数量 |
| --- | --- |
| sequential_single | 16 |
| torch_superbatch | 32 |
| triton_superbatch | 32 |

结论：

1. 当前 same-model phase-locked cohort 场景下，最关键的不是 generic MPS
2. 主要收益已经被 super-batch execution 吃掉
3. 这说明 VLA-aware cohort formation + same-model super-batch execution 才是主问题

---

## 5. Profiler：收益主要来自 launch/call amortization

代码：

- `src/gr00t/eval/profile_gr00t_superbatch_execution.py`

结果：

- `results/_tmp_gr00t_superbatch_execution_profile.json`

关键数字：

| mode | total self cuda | total cuda | total calls |
| --- | --- | --- | --- |
| sequential_single | 75886 us | 114218 us | 56169 |
| torch_superbatch | 29109 us | 43235 us | 13872 |
| triton_superbatch | 29058 us | 43169 us | 13890 |

结论：

1. 调用数下降大约 4 倍
2. 总 CUDA 时间明显下降
3. 热点仍然主要在 GEMM / attention
4. 因此当前贡献不是“某个小 kernel 神奇提速”，而是执行形态本身变了

---

## 6. 接回真实 policy path 后，super-batch 仍然有效

代码：

- `src/gr00t/eval/bench_gr00t_policy_prepared_superbatch_replay.py`
- `/root/autodl-tmp/Isaac-GR00T/gr00t/policy/gr00t_policy.py`
- `/root/autodl-tmp/Isaac-GR00T/gr00t/model/gr00t_n1d6/gr00t_n1d6.py`

说明：

真实 policy path 已经被拆成：

- `prepare_inference_inputs`
- `merge_prepared_inputs`
- `predict_normalized_action`
- `decode_normalized_action`

这样可以把“高层输入准备”和“执行边界优化”分开评估。

结果：

- `results/_tmp_gr00t_policy_prepared_superbatch_replay_breakdown_v3.json`

### 6.1 各模式阶段拆分

| mode | prepare p50 | predict p50 | decode p50 | service p50 | latency p50 | queue p50 |
| --- | --- | --- | --- | --- | --- | --- |
| request_seq_full | 11.168 ms | 206.980 ms | 1.852 ms | 220.001 ms | 321.099 ms | 101.098 ms |
| prepared_seq | 9.962 ms | 186.776 ms | 2.249 ms | 198.987 ms | 277.991 ms | 79.005 ms |
| prepared_superbatch | 9.498 ms | 99.995 ms | 1.276 ms | 110.769 ms | 146.887 ms | 36.118 ms |
| batch_prepare_prepared | 11.214 ms | 144.068 ms | 1.598 ms | 156.880 ms | 226.579 ms | 69.698 ms |
| raw_batch_full | 13.207 ms | 126.515 ms | 1.596 ms | 141.318 ms | 195.465 ms | 54.147 ms |

### 6.2 结论

1. `prepared_superbatch` 依然是当前真实 policy path 里最强的模式
2. 真实链路里的主要瓶颈仍然是 `predict`，不是 `prepare` 或 `decode`
3. `prepare` 只在大约 `10~13 ms`
4. `decode` 只在 `1~2 ms`
5. 真正重的是 backbone + action head + 它们之间的执行路径

这说明下一步真正该优化的是：

- action-head diffusion loop
- backbone/action-head 之间的 execution boundary
- same-model super-batch 的真实 runtime 接入

---

## 7. 当前能稳妥声称什么

截至目前，可以稳妥声称：

1. 对 GR00T N1.6 这类 chunked-action VLA，kernel-level 的正确方向不是 generic queue batching，而是 VLA-aware 的 same-model super-batch execution
2. 当 legal replan window 可以把请求对齐成 cohort 时，execution primitive 本身就足以决定 deadline 是否能守住
3. 当前收益主要来自对 repeated kernel launches 和 per-request 调度开销的摊销
4. 在当前 same-model 场景中，MPS 不是主要矛盾，super-batch execution 才是主要矛盾
5. 接回真实 policy path 后，`predict` 仍然是主瓶颈，因此后续优化必须继续深入 action head / backbone 边界

---

## 8. 当前还不能过度声称什么

截至目前，还不能过度声称：

1. 不能说已经做出了稳定优于 PyTorch 的最终 Triton kernel 主方案
2. 不能说已经把这条 primitive 完整接入真实多模型 online runtime 并完成闭环
3. 不能说当前收益完全来自 kernel 本身，而不是 phase-lock 调度与 execution primitive 的协同
4. 不能说 decode 或 collator 已经被彻底优化到位

---

## 9. 当前未完成但最值得继续挖的方向

下一步最值得继续做的是：

1. 继续优化 action head 的 diffusion inference hot loop
2. 减少 loop 内部的临时张量分配与重复拼接
3. 在真实 policy replay 下验证 `predict` 是否继续下降
4. 最终把这条 primitive 接回真实 multi-model serving runtime

一句话总结：

**当前最强的 GPU kernel level 主线，不是单个小 kernel，而是把 VLA 的相位可控性转成 same-model super-batch execution primitive；它已经被 microbench、replay、runtime ablation 和 profiler 四条证据链支持。**
