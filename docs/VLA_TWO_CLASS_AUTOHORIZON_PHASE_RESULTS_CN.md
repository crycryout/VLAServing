# 当前二类 AutoHorizon + 主动相位控制下的 Pi05 / GR00T N1.6 实验总结

## 1. 这份文档的范围

这份文档只总结当前主线配置下的实验结果：

- `AutoHorizon` 被收敛成两类语义
- 系统允许在合法窗口内主动控制相位
- 关注 `Pi05` 和 `GR00T N1.6` 两条工作负载

这里不讨论：

- 旧的非两类 `Pi05` 语义
- 模型压缩 / 去重方案

## 2. 当前的二类 AutoHorizon 语义

### 2.1 Pi05

`Pi05` 当前只保留两类 horizon：

- 所有 `<25` 的采样值都视为 `25`
- 所有 `>25` 的采样值都视为 `50`

因此 `Pi05` 的控制语义可以写成：

- horizon 目标集合：`{25, 50}`
- 合法提前重规划窗口：`[25, 50]`

含义是：

- 如果当前 chunk 还没耗尽，只要已经消耗了至少 `25` 个 action，就允许主动提前触发下一次推理

### 2.2 GR00T N1.6

`GR00T N1.6` 当前的实现也可以看成两类语义：

- 如果采样 horizon `<= 8`，则视为固定 `8`
- 如果采样 horizon `> 8`，则允许在 `[8, 16]` 内任意 action 位置提前触发

代码上对应的是：

- `horizon <= 8 -> allowed_range = [8, 8]`
- `horizon > 8 -> allowed_range = [8, 16]`

因此 `GR00T N1.6` 的当前语义不是简单的“只有一个 floor=8”，而是：

- `紧急类`
  - 必须在 `8` 这个位置触发
- `可移动类`
  - 可以在 `[8,16]` 内主动调相位

## 3. Pi05 的实验结果

### 3.1 配置

当前 `Pi05` 主线配置是：

- `30Hz -> 30hz_official_ft`
- `20Hz -> 20hz_quantiles`
- `10Hz -> 10hz_a_logits`
- `10Hz -> 10hz_b_autoh`
- GPU 上采用 `3-shell`
- 使用 predictive residency / prefetch
- 在 `[25, 50]` 内允许主动调相位

### 3.2 fixed-4 结果

在固定四机器人场景下，三种策略结果几乎一致：

| 策略 | p95 推理时延 | fleet score | min robot score | miss_autohorizon_ratio | hard miss | reply-over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_horizon` | `43.2133 ms` | `1.0000` | `1.0000` | `0.0000` | `0` | `0` |
| `phase_shift` | `43.2133 ms` | `1.0000` | `1.0000` | `0.0000` | `0` | `0` |
| `batch_align` | `43.2133 ms` | `1.0000` | `1.0000` | `0.0000` | `0` | `0` |

结论很直接：

- 在固定四机器人场景下，`Pi05` 已经稳定
- 主动调相位不会继续把时延压得更低
- 当前系统的瓶颈不在 fixed-4 的相位排布上

### 3.3 admission 结果

在 admission 场景下，三种策略也几乎一致：

| 策略 | mean admitted total | fleet score | min robot score | miss_autohorizon_ratio | phase shift abs actions | p95 推理时延 | hard miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_horizon` | `32.6667` | `0.9934` | `0.9644` | `0.0420` | `0.1409` | `43.2084 ms` | `0` |
| `phase_shift` | `32.6667` | `0.9933` | `0.9644` | `0.0428` | `0.1422` | `43.2084 ms` | `0` |
| `batch_align` | `32.6667` | `0.9934` | `0.9644` | `0.0420` | `0.1409` | `43.2084 ms` | `0` |

结论是：

- `Pi05` 当前主线收益来自更宽的 `[25, 50]` 合法窗口
- 不是来自更激进的主动相位移动
- 在这版配置里，phase control 的主要价值是把 `reservation + prefetch` 做得更容易，而不是直接提升 admitted robot 数

## 4. GR00T N1.6 的实验结果

### 4.1 配置

当前 `GR00T N1.6` 主线配置是：

- `30Hz -> 30hz_bridge`
- `20Hz -> 20hz_fractal`
- `10Hz -> 10hz_libero`
- `10Hz -> 10hz_rel30k`
- chunk size `16`
- `horizon_floor = 8`
- 如果 `horizon > 8`，允许在 `[8,16]` 内主动调相位
- 结构上使用 `shared-prefix residency`

### 4.2 仅用通用 phase policy 时的结果

如果只用通用的二类 horizon + 相位策略，而不引入 `GR00T` 专门的 same-model phase-lock runtime，那么结果如下。

#### fixed-4

| 策略 | p95 推理时延 | fleet score | min robot score | miss_autohorizon_ratio | phase shift abs actions | hard miss | reply-over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_horizon` | `43.8806 ms` | `0.8521` | `0.7242` | `0.1120` | `0.8114` | `0` | `0` |
| `phase_shift` | `43.8806 ms` | `0.8990` | `0.8066` | `0.1253` | `0.5904` | `0` | `0` |
| `batch_align` | `43.8806 ms` | `0.8991` | `0.8068` | `0.1264` | `0.5930` | `0` | `0` |

#### admission

| 策略 | mean admitted total | fleet score | min robot score | miss_autohorizon_ratio | phase shift abs actions | p95 推理时延 | hard miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_horizon` | `4.0` | `0.8568` | `0.7300` | `0.1140` | `0.7896` | `43.8806 ms` | `0` |
| `phase_shift` | `4.0` | `0.8975` | `0.8033` | `0.1216` | `0.5957` | `43.8806 ms` | `0` |
| `batch_align` | `4.0` | `0.8975` | `0.8033` | `0.1216` | `0.5957` | `43.8806 ms` | `0` |

这里的结论是：

- 二类 horizon + 通用 phase control 对 `GR00T` 的确会改善 `fleet score / min robot score`
- 但它并没有把 admission 容量从 `4` 继续提升
- 同时 `miss_autohorizon_ratio` 反而略高

也就是说：

- 对 `GR00T N1.6`，泛化的主动调相位不是主要收益来源
- 它只能改善质量，不能直接扩容

### 4.3 引入 GR00T 专门 runtime 后的结果

`GR00T N1.6` 真正有效的做法是：

- `shared-prefix residency`
- `same-model phase-lock batching`
- `quota-fair admission`

在这套 runtime 下，结果明显更强。

#### phase-lock batching

| 场景 | strict horizon | phase-lock batch | 结论 |
| --- | --- | --- | --- |
| `8` 机器人 | `mean_batch_size = 1.3845`，`reply_over = 7`，`p95 = 47.6112 ms` | `mean_batch_size = 2.0`，`reply_over = 0`，`p95 = 47.6112 ms` | `phase-lock` 把 `reply_over` 从 `7` 降到 `0` |
| `16` 机器人 | `mean_batch_size = 2.1966`，`reply_over = 15`，`p95 = 58.4292 ms` | `mean_batch_size = 4.0`，`reply_over = 0`，`p95 = 58.4292 ms` | `phase-lock` 把 `reply_over` 从 `15` 降到 `0` |

#### 稳定 serving

| 场景 | 结果 |
| --- | --- |
| `4` 机器人 | `43.8806 ms p95`，`reply_over = 0`，稳定 |
| `16` 机器人 | `58.4292 ms p95`，`batch = 4.0`，`reply_over = 0`，稳定 |

#### fair admission

相对 greedy batch-first admission：

- `accept-rate gap: 0.2921 -> 0.0994`
- `final-count gap: 18 -> 2`
- `mean p95: 90.0133 ms -> 58.4292 ms`

代价是：

- `mean_final_robot_count: 22.83 -> 15.67`

这说明：

- `GR00T` 上真正有效的相位控制形式，是把相同模型机器人主动锁相形成 cohort
- 而不是对每个机器人做自由 phase shift

## 5. 把两者放在一起看

### Pi05

当前最重要的是：

- 把控制语义压成 `{25, 50}`
- 允许在 `[25, 50]` 内合法提前请求
- 用更宽的窗口去服务 residency / prefetch

因此 `Pi05` 的 phase control 更像：

- `window-aware phase movement`

### GR00T N1.6

当前最重要的是：

- 把 horizon 分成 `<=8` 和 `>8` 两类
- 对 `>8` 的请求在 `[8,16]` 内主动调相位
- 但调相位的目标不是随便平移，而是同模型 cohort 化

因此 `GR00T N1.6` 的 phase control 更像：

- `cohort-aware phase lock`

## 6. 当前最稳的总结合并版

可以收敛成下面两句：

- 对 `Pi05`，二类 `AutoHorizon` 加主动相位控制是有效的，但主要收益来自更宽的 `[25,50]` legal replan window，本身不会继续显著提升 fixed-4 或 admission 容量。
- 对 `GR00T N1.6`，二类 horizon + 通用 phase shift 只能改善分数，真正决定性的方法是 `shared-prefix + same-model phase-lock batching + fair admission`。

## 7. 对应脚本与结果文件

### Pi05

- `src/bench_pi05_vla_serving_autoh25_50_phase_shift.py`
- `results/pi05_vla_serving_autoh25_50_phase_shift_20260413.json`

### GR00T N1.6

- `src/bench_vla_gpu_virtualization_policy_horizon_floor.py`
- `results/vla_gpu_virtualization_policy_horizon_floor_20260412.json`
- `results/vla_gpu_virtualization_policy_gr00t_batch_align_floor8_20260412.json`
- `src/gr00t/eval/bench_gr00t_shared_prefix_phase_lock_batch_mps.py`
- `results/gr00t_shared_prefix_phase_lock_batch_mps_20260412.json`
- `src/gr00t/eval/bench_gr00t_batch_only_fair_admission.py`
- `results/gr00t_batch_only_fair_admission_20260412.json`
