# GR00T N1.6 Mirage 完整端到端推理状态（2026-04-23）

## 1. 这次做了什么

这次不是再跑 `2-layer twoblock subgraph`，而是把 Mirage 接进了 **真实的 GR00T N1.6 端到端推理路径**：

- `synthetic observation`
- `processor / collate`
- `backbone (Eagle VLM)`
- `Mirage-backed action head`
- 输出 `action_pred`

对应脚本：

- [bench_gr00t_mpk_full_e2e_runtime.py](/root/autodl-tmp/VLAServing/src/gr00t/eval/bench_gr00t_mpk_full_e2e_runtime.py)

结果文件：

- [gr00t_mpk_full_e2e_runtime_20260423.json](/root/autodl-tmp/VLAServing/results/gr00t_mpk_full_e2e_runtime_20260423.json)

这里的 `Mirage-backed action head` 口径是：

- 使用官方 `processor`
- 使用官方 `backbone`
- 使用 Mirage 执行 **完整 32-layer DiT core**
- 保留 `action_encoder / action_decoder / final norm+proj` 在 PyTorch 中

所以这已经是一个**真实可运行的端到端推理路径**，不是之前的 toy subgraph。

## 2. 为了让完整路径跑起来，修了什么

在直接跑完整 32-layer Mirage core 时，Mirage runtime 的 worker queue 上限不够，触发了 device-side assert。

对应本地修复：

- `/root/autodl-tmp/mirage/include/mirage/persistent_kernel/persistent_kernel.cuh`

修改内容：

- `per_worker_queue_len: 1024 -> 8192`
- `per_sched_queue_len: 1024 -> 8192`

这一步的作用不是优化性能，而是**让完整 32-layer GR00T core 能够实际被 Mirage runtime 执行**。

## 3. 实验设置

输入使用可复现的 synthetic observation：

- video key: `ego_view_bg_crop_pad_res256_freq20`
- video shape: `[1, 1, 256, 256, 3]`
- state dims:
  - `left_arm = 7`
  - `right_arm = 7`
  - `left_hand = 6`
  - `right_hand = 6`
  - `waist = 3`

这次实际跑出来的 prepared boundary 形状是：

- `backbone_features`: `[1, 109, 2048]`
- `backbone_attention_mask`: `[1, 109]`
- `image_mask`: `[1, 109]`
- `action_state`: `[1, 1, 128]`
- `action_pred`: `[1, 50, 128]`

官方对照口径：

- `torch.compile` 只作用在 `action_head.model.forward`
- 这和官方 `benchmark_inference.py` 的 compile 口径一致

Mirage 口径：

- `processor + backbone + Mirage-backed single 32-layer online_notoken action-head core`

## 4. 关键结果

### 4.1 Mirage 编译时间

- Mirage full core compile time: `21063.67 ms`

也就是大约：

- `21.06 s`

### 4.2 官方 compiled E2E

- data processing mean: `6.05 ms`
- backbone mean: `140.22 ms`
- action head mean: `32.07 ms`
- e2e mean: `186.70 ms`

### 4.3 Mirage-backed E2E

- data processing mean: `6.41 ms`
- backbone mean: `139.27 ms`
- action head mean: `207334.62 ms`
- e2e mean: `207480.72 ms`

也就是：

- Mirage action head 约 `207.33 s`
- Mirage E2E 约 `207.48 s`

### 4.4 数值对齐

和官方 compiled action head 对比：

- `max_abs = 159.109375`
- `mean_abs = 2.9971513748168945`
- `official_sum = -1264.4436`
- `mirage_sum = 124.8673`

这说明当前这条完整 Mirage 路径虽然**能跑通**，但**数值还没有闭合**。

## 5. 结论

当前结论必须严格分成两部分。

### 5.1 成功的部分

已经成功证明：

- Mirage 不再只是 `mini twoblock` 子图
- GR00T N1.6 已经存在一条 **真实可执行的 Mirage-backed 完整端到端推理路径**
- 这条路径可以从 observation 一直走到 `action_pred`

所以如果问题是：

- “Mirage 的完整端到端推理目前有没有实现出来？”

答案现在是：

- **有，已经实现并跑通了**

### 5.2 还没有成功的部分

如果问题是：

- “Mirage 这条完整端到端推理是否已经优于官方 compiled 推理栈？”

答案仍然是：

- **没有**

原因很直接：

- 速度还差了几个数量级
- 数值误差也还很大

也就是说，当前状态是：

- **功能性成功**
- **性能失败**
- **数值闭合失败**

## 6. 下一步最值得做什么

现在最值得继续的不是再证明“能不能跑”，这件事已经证明了。

真正该继续的是：

1. 先把 `linear_generic` 路径替换掉
   - 当前完整 Mirage 路径几乎肯定仍然被 generic linear 主导

2. 再定位 full core 的数值误差来源
   - 先做 block-level diff
   - 再做 step-level diff
   - 最后定位是 attention、AdaLN、还是 residual/FF 链路累计误差

3. 只有在数值闭合后，才值得继续追求
   - `Mirage E2E < official compiled E2E`

## 7. 一句话总结

**Mirage 的 GR00T N1.6 完整端到端推理已经实现并真实跑通；但当前版本既不够快，也还没有数值对齐，因此它现在是“可运行原型”，不是“可用替代栈”。**
