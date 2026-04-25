# GR00T MultiStage Persistent/Graph Runtime 总结

更新时间：2026-04-25

## 1. 背景与目标

这轮工作的目标不是再讨论 `MicroBatch` 是否有效，而是继续沿着 `MultiStage` 的低层优化方向，把同卡并发带来的时间开销尽量打掉。核心问题是：

- `naive multistage overlap` 会把 latency 打爆
- 之前已经证明 `DiT step-level microbatch` 本身有效，但 `VLM` 的 staged overlap 不稳定
- 需要判断更低层的 `persistent/kernel/cudagraph` 路线是否真的能改善端到端结果

这轮工作的范围包括：

- `DiT step` 的 `CUDA Graph` 执行器
- `VLM postprefill` 的 `CUDA Graph` 执行器
- `SigLIP2` 固定形状 fast path
- `full VLM graph executor`
- 把这些路径接回 unified async replay，观察真实 end-to-end 变化

## 2. 方法概览

### 2.1 Step Graph

先把 `DiT step batch` 固定形状化并做 `CUDA Graph` capture。这个路径只覆盖：

- `action_encoder`
- `DiT model`
- `action_decoder`

实现见：

- [src/gr00t/eval/gr00t_step_cudagraph.py](../src/gr00t/eval/gr00t_step_cudagraph.py)
- [src/gr00t/eval/bench_gr00t_dit_step_cudagraph_executor.py](../src/gr00t/eval/bench_gr00t_dit_step_cudagraph_executor.py)

### 2.2 VLM Postprefill Graph

由于完整 `VLM` 早期 capture 会被 `SigLIP2` 和 `Qwen3` 的 capture-unsafe 路径阻塞，所以先只把：

- `llm_body`
- `action_head.vlln`
- `state_encoder`

做成 `postprefill` graph。

实现见：

- [src/gr00t/eval/gr00t_vlm_prepare_cudagraph.py](../src/gr00t/eval/gr00t_vlm_prepare_cudagraph.py)
- [src/gr00t/eval/bench_gr00t_vlm_prepare_cudagraph_executor.py](../src/gr00t/eval/bench_gr00t_vlm_prepare_cudagraph_executor.py)

### 2.3 SigLIP2 Fixed-Shape Patch

为了继续推进 `vision prefill` 路线，新增了 `SigLIP2` 固定形状 patch：

- 缓存 window reorder permutation
- 缓存 rope freqs
- 缓存 packed FA2 的 `cu_seqlens`

这样 `SigLIP2` 的窗口重排和 packed flash-attn 不再在 forward 中动态构造索引和 `cu_seqlens`。

实现见：

- [src/gr00t/eval/gr00t_siglip2_fixedshape.py](../src/gr00t/eval/gr00t_siglip2_fixedshape.py)
- [src/gr00t/eval/bench_gr00t_siglip2_fixedshape_probe.py](../src/gr00t/eval/bench_gr00t_siglip2_fixedshape_probe.py)

### 2.4 Full VLM Graph

在 fixed-shape `SigLIP2` 基础上，进一步实现 `full VLM graph executor`，将：

- `vision_model`
- `pixel_shuffle_back`
- `projector`
- image-token fuse
- `llm_body`
- `action_head.vlln`
- `state_encoder`

合并为一条 graph 路径。

实现见：

- [src/gr00t/eval/gr00t_vlm_full_cudagraph.py](../src/gr00t/eval/gr00t_vlm_full_cudagraph.py)
- [src/gr00t/eval/bench_gr00t_vlm_full_cudagraph_executor.py](../src/gr00t/eval/bench_gr00t_vlm_full_cudagraph_executor.py)

### 2.5 Unified Async Integration

上述 graph 路径都已接入 unified runtime：

- [src/gr00t/eval/bench_gr00t_unified_multistage_microbatch_runtime.py](../src/gr00t/eval/bench_gr00t_unified_multistage_microbatch_runtime.py)

用于比较两条端到端路线：

- `step graph + postprefill graph`
- `step graph + full VLM graph`

## 3. 关键结果

### 3.1 Step Graph 有明确收益

结果文件：

- [results/_tmp_gr00t_dit_step_cudagraph_executor.json](../results/_tmp_gr00t_dit_step_cudagraph_executor.json)

关键点：

- `batch=1` 单 step `p50 17.34 -> 4.57 ms`
- `batch=1` 4-step loop `p50 73.88 -> 17.55 ms`
- `batch=2` 单 step `p50 17.69 -> 5.65 ms`
- `batch=4` 单 step `p50 18.11 -> 7.35 ms`

结论：

- `DiT step` 的 launch/runtime overhead 是显著的
- 这条 `cudagraph` 路线在 `DiT` 上非常有效

### 3.2 Async 最优端到端仍然是 Step + Postprefill

结果文件：

- [results/_tmp_gr00t_unified_async_multigraph_compare.json](../results/_tmp_gr00t_unified_async_multigraph_compare.json)

当前最优变体是 `step_and_vlm`：

- `request p50 = 868.69 ms`
- `request p95 = 1564.29 ms`
- `makespan = 2214.37 ms`
- `vlm_service p50 = 85.32 ms`
- `dit_step_service p50 = 5.55 ms`
- `vlm host_overhead p50 = 0.45 ms`

对比 baseline：

- baseline `p50 = 1577.58 ms`
- baseline `p95 = 2682.02 ms`

结论：

- `step graph` 是最大杠杆
- `postprefill graph` 进一步压掉了 handoff / host overhead
- 但这条路径仍然不是 deadline-safe

### 3.3 Fixed-Shape SigLIP2 本身就能压低 Vision Eager

结果文件：

- [results/_tmp_gr00t_siglip2_fixedshape_probe.json](../results/_tmp_gr00t_siglip2_fixedshape_probe.json)

关键点：

- `SigLIP2 eager 54.91 -> 43.15 ms`
- `last_hidden_state_max_abs = 0`

结论：

- 即使不看 graph capture，固定形状 `SigLIP2` fast path 本身就有收益
- 语义上与原始路径一致

### 3.4 Full VLM Graph 的局部 steady-state 是正收益

结果文件：

- [results/_tmp_gr00t_vlm_full_cudagraph_executor.json](../results/_tmp_gr00t_vlm_full_cudagraph_executor.json)

关键点：

- `prepare_task baseline p50 = 67.89 ms`
- `prepare_task full_graph p50 = 44.10 ms`
- `backbone_mean_abs = 0.0166`
- `state_max_abs = 0`

结论：

- 从局部 steady-state 看，`full VLM graph` 是有效的
- 数值漂移主要来自 `llm` 使用 `sdpa` graph backend，而不是 `vision` fixed-shape patch 本身

### 3.5 Full VLM Graph 接回 Async Replay 后没有赢

结果文件：

- [results/_tmp_gr00t_unified_async_fullgraph_single.json](../results/_tmp_gr00t_unified_async_fullgraph_single.json)

关键点：

- `vlm_service p50 = 47.85 ms`
- 但 `request p50 = 1140.01 ms`
- `request p95 = 1518.44 ms`
- `queue_wait p50 = 950.35 ms`
- `dit_step_service p50 = 25.42 ms`

和当前最优 `step_and_vlm` 对比：

- `VLM` 自身更快：`85.32 -> 47.85 ms`
- 但 `DiT step` 明显更慢：`5.55 -> 25.42 ms`
- 所以 end-to-end 反而更差：`868.69 -> 1140.01 ms`

结论：

- `full VLM graph` 并没有成为新的最优路径
- 它把 `VLM` 压快了，但放大了和 `DiT step` 的 GPU 内部资源争抢

## 4. 根因分析

这轮工作把问题收敛得比较清楚了。

### 4.1 Host / Sync / Handoff 已经不是主瓶颈

在 `step + postprefill graph` 最优路径下：

- `vlm host_overhead p50 = 0.45 ms`

在 `step + fullgraph` 路径下：

- `vlm host_overhead p50 = 0.62 ms`

说明：

- 继续优化 host-side orchestration，收益空间已经很小

### 4.2 剩余问题是 GPU 内部资源干扰

`fullgraph` 证明了一件事：

- 单独看 `VLM`，graph fusion 确实有效
- 但一旦放进统一 replay，`DiT step` latency 会被拖高

这意味着当前系统剩余的大头不是：

- launch overhead
- stage handoff
- CPU runtime

而是：

- `VLM` 和 `DiT` 在同卡上的 Tensor Core / HBM / cache 争抢

## 5. 最终结论

这轮工作之后，当前最好的结论是：

1. `persistent/cudagraph` 路线在局部组件上是有效的
2. `DiT step graph` 是当前最大的稳定收益来源
3. `postprefill graph` 是当前最优的 `VLM` graph 形态
4. `full VLM graph` 虽然压低了局部 `VLM` latency，但没有改善 unified async end-to-end
5. 当前这套本地优化已经把 `MultiStage` 的 host/runtime overhead 基本打到很低，剩余主问题是 GPU 内部资源隔离

因此，当前 repo 内最值得保留的最优路径是：

- `step graph + postprefill graph`

而不是：

- `step graph + full VLM graph`

## 6. 建议的下一步

如果继续往下做，优先级应该是：

1. `executor pool` 或多实例 graph，验证受控多实例是否能减轻 queue buildup
2. 更硬的 GPU 资源分区，而不是继续堆 graph fusion
3. 显式控制 `VLM` 与 `DiT` 的 overlap，而不是让它们自由竞争

不建议继续投入的方向：

1. 继续扩大 host-side 调度逻辑复杂度
2. 继续把更多小 stage 单独 graph 化
3. 仅靠更多 graph fusion 来期待自动改善 end-to-end

## 7. 相关文件索引

核心代码：

- [src/gr00t/eval/gr00t_step_cudagraph.py](../src/gr00t/eval/gr00t_step_cudagraph.py)
- [src/gr00t/eval/gr00t_vlm_prepare_cudagraph.py](../src/gr00t/eval/gr00t_vlm_prepare_cudagraph.py)
- [src/gr00t/eval/gr00t_siglip2_fixedshape.py](../src/gr00t/eval/gr00t_siglip2_fixedshape.py)
- [src/gr00t/eval/gr00t_vlm_full_cudagraph.py](../src/gr00t/eval/gr00t_vlm_full_cudagraph.py)
- [src/gr00t/eval/bench_gr00t_unified_multistage_microbatch_runtime.py](../src/gr00t/eval/bench_gr00t_unified_multistage_microbatch_runtime.py)

核心结果：

- [results/_tmp_gr00t_dit_step_cudagraph_executor.json](../results/_tmp_gr00t_dit_step_cudagraph_executor.json)
- [results/_tmp_gr00t_unified_async_multigraph_compare.json](../results/_tmp_gr00t_unified_async_multigraph_compare.json)
- [results/_tmp_gr00t_siglip2_fixedshape_probe.json](../results/_tmp_gr00t_siglip2_fixedshape_probe.json)
- [results/_tmp_gr00t_vlm_full_cudagraph_executor.json](../results/_tmp_gr00t_vlm_full_cudagraph_executor.json)
- [results/_tmp_gr00t_unified_async_fullgraph_single.json](../results/_tmp_gr00t_unified_async_fullgraph_single.json)

背景文档：

- [GR00T_MULTISTAGE_MICROBATCH_SUMMARY_20260423_CN.md](./GR00T_MULTISTAGE_MICROBATCH_SUMMARY_20260423_CN.md)
- [GR00T_STAGE_PARTITION_AND_MICROBATCH_RESULTS_20260423_CN.md](./GR00T_STAGE_PARTITION_AND_MICROBATCH_RESULTS_20260423_CN.md)
- [GR00T_KERNEL_LEVEL_VLA_OPTIMIZATION_20260414.md](./GR00T_KERNEL_LEVEL_VLA_OPTIMIZATION_20260414.md)
