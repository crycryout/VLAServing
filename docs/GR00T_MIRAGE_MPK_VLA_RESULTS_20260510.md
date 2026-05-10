# GR00T N1.6 VLA Mirage / MPK Results

本文档记录截至 2026-05-10 的 GR00T N1.6 VLA Mirage / MPK 实验结果。

核心结论是：

1. 当前还没有完成完整的 `VLM + DiT` MPK mega-kernel。
2. 已经证明 VLA 推理栈里存在可被 MPK-style data plane 利用的 kernel boundary / launch / timeline gap。
3. VLM batch=1 的 fixed-shape graph replay 已经能显著降低同数学路径的 timeline gap，是当前最强的 VLM 侧证据。
4. DiT hand-MPK MVP 已经接入真实 single-step 路径，但当前性能仍明显差于 CUDA Graph step。
5. Mirage/MPK attention bridge 已经能编译、launch，并且直接 helper correctness 基本闭合，但还不是最终高性能 Tensor Core task body。
6. Serving-stage scheduler MVP 证明了 stage credit、phase-local deadline、request-path service curve 等控制面策略能缓解部分 backlog，但在压力负载下仍存在 fairness-capacity tradeoff。

## Artifact Index

主要结果文件：

- `results/gr00t_official_compile_benchmark_20260510.json`
- `results/gr00t_official_mpk_potential_graphproxy_20260510.json`
- `results/gr00t_vlm_mpk_dataplane_proxy_20260510.json`
- `results/gr00t_dit_single_step_hand_mpk_mvp_cutlass_linear_tiled_attention_auto_profile_20260510.json`
- `results/gr00t_mpk_bridge_attention_correctness_20260510.json`
- `results/gr00t_mpk_bridge_attention_self_padded64_launch_20260510.json`
- `results/gr00t_mpk_bridge_attention_cross_padded64x256_launch_20260510.json`
- `results/gr00t_mpk_stage_scheduler_mvp_20260510.json`
- `results/gr00t_mpk_full_e2e_runtime_20260423.json`

相关脚本在本地 workspace 中主要位于：

- `src/gr00t/eval/bench_gr00t_official_mpk_potential.py`
- `src/gr00t/eval/bench_gr00t_vlm_mpk_dataplane_proxy.py`
- `src/gr00t/eval/bench_gr00t_dit_single_step_hand_mpk_mvp.py`
- `src/gr00t/eval/bench_gr00t_mpk_stage_scheduler_mvp.py`
- `src/gr00t/eval/probe_gr00t_mpk_bridge_attention_correctness.py`
- `src/gr00t/eval/probe_gr00t_mpk_task_compile_dit_attention.py`

注意：Mirage/MPK bridge 的一部分实现依赖 `/root/autodl-tmp/mirage` 的本地 patch，不只是在 `VLAServing` 仓库内完成。

## Official Compile Baseline

对应结果：

- `results/gr00t_official_compile_benchmark_20260510.json`

配置：

- device: `NVIDIA GeForce RTX 4090`
- model: GR00T N1.6
- dtype: bf16
- inference steps: 4
- baseline: official-style `torch.compile`

`torch.compile` p50 结果：

| Component | p50 latency |
| --- | ---: |
| data processing | `5.60 ms` |
| backbone / VLM | `32.26 ms` |
| action head / DiT | `16.69 ms` |
| E2E | `55.10 ms` |

这个结果是后续 MPK 论证的强 baseline。任何 MPK 结果都应该和同 scope 的 official compiled / CUDA Graph 路径比较，而不是和 eager PyTorch 比较。

## MPK Potential Proxy

对应结果：

- `results/gr00t_official_mpk_potential_graphproxy_20260510.json`

官方 compiled 近似端到端 p50：

- `53.63 ms`

profile 中统计到的 model CUDA kernel sum：

- `23.64 ms`

理想化 MPK lower bound：

- data processing p50: `6.27 ms`
- model CUDA kernel sum: `23.64 ms`
- ideal E2E lower bound: `29.91 ms`

这说明当前 VLA 推理栈中存在显著的可移除 timeline gap。这个实验不是已经实现的 MPK 加速结果，而是证明：

如果 MPK-style mega-kernel / persistent data plane 能保留现有 kernel 的数学效率，同时消除足够多的 kernel boundary / launch / scheduling gap，那么单个 VLA inference 理论上可以低于官方 compiled 栈。

## VLM Data Plane Proxy

对应结果：

- `results/gr00t_vlm_mpk_dataplane_proxy_20260510.json`

### Same-Math Fixed-Shape VLM Forward

| Mode | CUDA p50 |
| --- | ---: |
| uncaptured fixed-shape | `26.78 ms` |
| full graph replay | `10.82 ms` |

关键指标：

- p50 speedup: `2.47x`
- p50 latency delta: `15.95 ms`
- math kernel sum ratio, graph over uncaptured: `0.967`
- timeline gap reduction: `25.47 ms`

解释：

这个实验保留了相同 VLM 数学路径和 fixed-shape resident buffer，只用 graph replay 移除了 host/operator launch boundaries。kernel work 基本没有减少，但 wall time 明显降低，因此它是当前最干净的 MPK-style data plane 证据。

### Serving Prepare Scope

| Mode | CUDA p50 |
| --- | ---: |
| baseline prepare task | `34.30 ms` |
| full graph launch + materialize | `10.85 ms` |

关键指标：

- p50 speedup: `3.16x`
- p50 latency delta: `23.45 ms`

结论：

VLM 侧不是没有 MPK 收益。相反，VLM 是当前最强的 data-plane gap 证据。但是这仍然是 CUDA Graph / graph replay proxy，不是完整 VLM MPK mega-kernel。合理路线是先做 `VLM postprefill graph / postprefill MPK`，不是直接做 full VLM graph 或 full VLM mega-kernel。

## DiT Single-Step Hand-MPK MVP

对应结果：

- `results/gr00t_dit_single_step_hand_mpk_mvp_cutlass_linear_tiled_attention_auto_profile_20260510.json`

| Mode | single-step latency |
| --- | ---: |
| official step | `17.88 ms` |
| CUDA Graph step | `4.73 ms` |
| current hand-MPK step | `190.07 ms` |

Correctness against official step:

| Comparison | max_abs | mean_abs |
| --- | ---: | ---: |
| CUDA Graph vs official | `0.0` | `0.0` |
| hand-MPK vs official | `0.1674` | `0.0378` |

结论：

当前 DiT hand-MPK 已经接入真实 single-step action head 路径，但它只是功能性 MVP，不是性能成功。主要问题仍然是 task body 质量不够，尤其是 small-M bf16 linear、attention、FFN、AdaLN/residual 等路径还没有替换成高性能 shape-specific Tensor Core task。

当前正确目标不是宣称 hand-MPK 已经更快，而是：

1. 先把 correctness drift 收敛；
2. 再用 Mirage/CUTLASS/Tensor Core task 替换 scalar/generic task；
3. 最后再和 `4.73 ms` 的 DiT CUDA Graph step 对比。

## Mirage / MPK Attention Bridge

对应结果：

- `results/gr00t_mpk_bridge_attention_correctness_20260510.json`
- `results/gr00t_mpk_bridge_attention_self_padded64_launch_20260510.json`
- `results/gr00t_mpk_bridge_attention_cross_padded64x256_launch_20260510.json`

当前已经实现的桥接能力：

1. GR00T padded attention helper 可以作为 MPK-callable task variant 被注册。
2. MPK task graph compiler 可以生成包含该 task 的 persistent-kernel 代码。
3. self-attention 和 cross-attention 两种 GR00T DiT shape 都能 compile 和 launch。
4. 直接 helper correctness 对齐 PyTorch scaled masked softmax reference。

### Direct Helper Correctness

| Case | Shape | run mean | max_abs first 51 q | mean_abs first 51 q |
| --- | --- | ---: | ---: | ---: |
| self attention | `heads=32, q_pad=64, kv_pad=64, valid_kv=51, head_dim=48, bf16` | `0.494 ms` | `0.00377` | `0.000262` |
| cross attention | `heads=32, q_pad=64, kv_pad=256, valid_kv=203, head_dim=48, bf16` | `1.917 ms` | `0.00193` | `0.000139` |

### MPK Task Compile / Launch Probe

| Case | compile | launch | launch mean |
| --- | --- | --- | ---: |
| self padded64 | ok | ok | `48.95 ms` |
| cross padded64x256 | ok | ok | `212.90 ms` |

解释：

这里的 compile / launch probe 证明的是 MPK task variant plumbing 已经打通，不应该把这个 test-mode launch latency 当成最终 task-only performance。当前 helper 是 correctness-first 标量/warp实现，不是最终 Mirage MMA / Tensor Core kernel body。

下一步必须做的是把 Mirage 生成的高性能 attention kernel body 改造成 MPK task variant 可调用的 device/helper 函数，并补齐 GR00T 的 scale、mask、padding slice 语义。

## Serving-Stage Scheduler MVP

对应结果：

- `results/gr00t_mpk_stage_scheduler_mvp_20260510.json`

这个实验是控制面模拟，不是完整 runtime。它用测得的 Prefill / LLM / DiT CUDA Graph latency 来模拟 MPK data plane 上的 stage-aware serving scheduler。

策略包含：

1. request-path service curve；
2. phase-local deadline；
3. stage credit，防止 Prefill 过量超前；
4. fairness-capacity tradeoff，通过 per-group DiT batch cap 控制；
5. DiT 按 denoising-step readiness 做 batch。

### Stable 31-Robot Scenario

mix:

- `10Hz: 16`
- `20Hz: 8`
- `30Hz: 7`

| Policy | completed | miss rate | p50 latency | p95 latency | mean DiT batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill-first naive | `67 / 67` | `1.000` | `1459.63 ms` | `1730.17 ms` | `7.44` |
| stage-credit EDF | `67 / 67` | `0.791` | `814.24 ms` | `1719.07 ms` | `1.46` |

变化：

- miss rate: `-0.209`
- p95 latency: `-11.10 ms`
- mean DiT batch: `-5.99`

结论：stage credit 和 phase-local EDF 能明显降低中位 latency 和 miss rate，但代价是 DiT batch 变小，吞吐型 batch 收益减少。

### Stress Scenario

mix:

- `10Hz: 24`
- `20Hz: 12`
- `30Hz: 10`

| Policy | completed | miss rate | p50 latency | p95 latency | mean DiT batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill-first naive | `98 / 98` | `1.000` | `2352.26 ms` | `2541.01 ms` | `7.54` |
| stage-credit EDF | `98 / 98` | `0.929` | `1549.40 ms` | `3131.25 ms` | `1.18` |

变化：

- miss rate: `-0.071`
- p95 latency: `+590.24 ms`
- mean DiT batch: `-6.36`

结论：压力负载下，stage-aware policy 能降低 miss rate 和 p50 latency，但 p95 latency 可能恶化。这证明 serving 层仍然有真实困难需要解决，不能只靠 naive MPS overlap 或简单 FIFO batching。

## Historical Full E2E Mirage Path

对应结果：

- `results/gr00t_mpk_full_e2e_runtime_20260423.json`

这条路径已经能把 Mirage 接到真实 GR00T N1.6 E2E 推理路径中：

- processor / collate；
- official backbone；
- Mirage-backed action-head core；
- output `action_pred`。

但是它仍然不是可用替代栈。

| Mode | action head | E2E |
| --- | ---: | ---: |
| historical official compiled path | `32.07 ms` | `186.70 ms` |
| Mirage-backed path | `207334.62 ms` | `207480.72 ms` |

Correctness:

- `max_abs = 159.1094`
- `mean_abs = 2.9972`

结论：

这条历史路径证明了 full E2E plumbing 存在，但速度和 correctness 都失败。它不能作为 MPK 加速成功证据，只能作为集成起点。

## Current Best Runtime Route

当前最值得保留的 runtime 路线仍然是：

1. `VLM batch=1 fixed-shape graph / postprefill graph`
2. `DiT microbatch + per-step CUDA Graph`
3. stage-aware serving scheduler
4. 后续逐步把 DiT attention / linear / FFN task 替换成 Mirage/MPK 生成的高性能 task body

不建议直接开始 full VLM mega-kernel，原因是：

1. full VLM operator 覆盖和 shape 管理复杂度太高；
2. CUDA Graph 已经能吃掉 VLM 侧大量 launch/timeline gap；
3. full VLM mega-kernel 的 correctness、workspace、memory residency、debugging 风险过大；
4. 当前 DiT task body 还没有达到 CUDA Graph step baseline，应该先把小 scope 做正确、做快。

## Limitations

当前结果不能支持这些说法：

1. 不能说完整 VLM + DiT MPK mega-kernel 已经实现。
2. 不能说 Mirage-backed full E2E 已经快于官方推理栈。
3. 不能把 MPK compile / launch probe 的 test-mode latency 当成最终 serving latency。
4. 不能把 VLM CUDA Graph proxy 直接等同于 full VLM MPK。

当前可以支持这些说法：

1. VLA 推理栈存在显著可移除 timeline gap。
2. VLM fixed-shape data plane 能在保留 math kernel work 的情况下显著减少 wall time。
3. DiT single-step hand-MPK 路径已经接入，但还需要高性能 task body。
4. Mirage/MPK task variant bridge 对 GR00T padded attention 已经能 compile、launch，并且 helper correctness 基本闭合。
5. Serving control plane 需要 stage credit、deadline、fairness-capacity tradeoff，而不是只做 naive overlap。

## Recommended Next Steps

优先级从高到低：

1. 把 GR00T DiT attention 的 correctness-first helper 替换成 Mirage/CUTLASS/Tensor Core 级别的 task body。
2. 补齐 small-M bf16 linear，尤其是 q_len 小、输出维度为 `1536 / 6144` 的 DiT 路径。
3. 做 block-level 和 step-level diff，定位 hand-MPK 与 official step 的数值误差来源。
4. 在 DiT single-step MPK 接近 CUDA Graph step 后，再接入 serving-stage scheduler 的真实 executor。
5. VLM 侧继续沿 postprefill graph / postprefill MPK 做，不直接做 full VLM mega-kernel。

## Final Conclusion

当前 MPK 工作已经从概念论证推进到了 task bridge 和 serving scheduler MVP，但还没有完成完整高性能 MPK runtime。

最强证据在 VLM data-plane proxy：同数学路径下，fixed-shape graph replay 将 VLM CUDA p50 从 `26.78 ms` 降到 `10.82 ms`，并且 math kernel sum ratio 仍为 `0.967`。这说明 MPK-style data plane 的目标是合理的。

最大短板在 DiT hand-MPK：当前 single-step hand-MPK 为 `190.07 ms`，仍远慢于 `4.73 ms` 的 CUDA Graph step，并且 correctness 还没有完全闭合。

因此当前正确表述是：

**MPK-style VLA data plane 的收益已经被实验支持，但完整 MPK mega-kernel 还处在实现和优化阶段。短期保留路线应是 VLM postprefill graph + DiT step CUDA Graph / microbatch，长期再用 Mirage/MPK 高性能 task 逐步替换 DiT 和 VLM postprefill 的关键子图。**
