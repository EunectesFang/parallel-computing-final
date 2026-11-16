# CAVC V3 实施与审核报告

**日期**：2026-11-15  
**仓库**：`C:\Users\Fang\Desktop\并行计算\期末项目说明\翻译`  
**分支**：`cavc-controller-v3`  
**代码 HEAD**：`4984b63`（计划文档提交 `4d1acbf`）  
**状态**：**可跑 Kaggle preflight 验证接线；尚不可用于 BLEU/LAAL/value-learning 结论。**

---

## 一页结论

| 项 | 结果 |
|---|---|
| 代码实现 | 7 个 Task 全部完成，`pfba-lab/week/controller_v3/` |
| 测试 | 169 个 controller 测试通过；`pfba-lab/week` 全套 175 通过 |
| CPU smoke | 9 targets、2 rollouts、checkpoint/summary/traces 正常；BLEU `unavailable`、COMET `skipped` |
| 逐 Task 审核 | 7/7 Approved；所有 Critical/Important 已修并复审 |
| 最终代码审核 | **0 Critical，6 Important，6 Minor**；结论 “Ready for preflight with fixes” |
| 建模一致性审核 | **Aligned with documented gaps**；一个结构性缺陷（shared `next_state`） |
| Kaggle kernel | 已生成并校验（20 个文件哈希一致），**未 push、未提交** |
| 数据集 | dev 训练 2422 句 / 调参 286 / 留出 295；test 干净确认从 `test:0140` 起 |
| Kaggle 预计 | preflight 0.5–1 h；200–300 句 1.5–2.5 h；全量 dev+7B 4–6 h；确认 1–2 h；合计 7–11 h |

**一句话**：架构已经从 one-hot 分类改成 value controller，泄漏边界、per-candidate 种子、cost id、post-controller 非提交语义、checkpoint 诚实性都做对了；但训练侧的 `History` 全零、标注轨迹从不选 READ、Bellman backup 共用 `next_state`，这三件事会让学到的价值函数偏掉。修完这三项再花 Kaggle 额度。

---

## 1. 交付物与验证

### 1.1 分支与提交

```text
branch: cavc-controller-v3
code HEAD: 4984b63  Fix Task 7 review round 2: untrained checkpoint honesty
docs HEAD: 4d1acbf  Add CAVC v3 implementation plan
base:      7aca8ec  (main)
```

`main` 未改动；所有代码在 feature branch。

### 1.2 模块清单

`pfba-lab/week/controller_v3/`

| 模块 | 职责 |
|---|---|
| `action.py` | `Action`、READ、9 个默认写动作、25 个扩展动作、6 维动作特征 |
| `config.py` | 冻结的 `ControllerConfig` |
| `features.py` | 34 维在线状态向量、`FeatureVector`、`StateInput`、History/Runtime/Confidence |
| `text_encoder.py` | 词表 `Vocab` + 双 GRU 文本编码器 |
| `network.py` | `CAVCNetwork`：状态编码器、READ 头、动作条件 WRITE 头 |
| `trainer.py` | Huber Q、pairwise ranking、BCE error/no-emit、Huber emit、目标网络 |
| `cost_model.py` | `MeasuredCostTable`、`PowerLawCostModel`、`PipelineCostModel` |
| `runtime.py` | `ControlledRuntime`：per-candidate 采样、translate、RALCP、vote 统计、`extend` |
| `reward.py` | quality / delay / terminal 奖励 |
| `labeler.py` | 反事实一步标签：同一状态、嵌套候选、oracle 前缀匹配 |
| `rollout.py` | 闭环 rollout：统一 READ/WRITE 打分、post-controller、virtual completion |
| `post_controller.py` | STOP / ADD_2 / WAIT（只提议，不提交） |
| `evaluator.py` | BLEU/COMET 状态透传、summarize、paired bootstrap |
| `targets.py` | `ActionOutcome` -> `ActionTarget`，非空 state_key |
| `checkpoint.py` | `weights_only=True`、格式版本、shape 校验、trained 标记 |
| `cli.py` | `--dry-run`、`--smoke` |
| `kaggle.py` | 真实适配器、cost calibration、preflight/dev/confirm |
| `README.md` | 架构、特征、动作、损失、泄漏边界、Kaggle 命令 |

### 1.3 验证命令与结果

```bash
"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests
# 169 passed in 16.22s

"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week
# 175 passed in 14.28s

cd pfba-lab/week
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --dry-run
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --smoke --output-dir <tmp>
# 9 targets, 2 rollouts, checkpoint/summary/traces written
# BLEU: unavailable, COMET: skipped
```

`compileall` 通过；`git diff --check` 通过；工作树除计划文档外干净。

### 1.4 Kaggle kernel

```text
pfba-lab/kaggle-runs/pfba-cavc-v3/kaggle_entry.py
pfba-lab/kaggle-runs/pfba-cavc-v3/kernel-metadata.json
```

元数据：

```json
{
  "id": "eunectesqaq/pfba-cavc-v3-preflight",
  "title": "pfba cavc v3 preflight",
  "code_file": "kaggle_entry.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

生成器 `pfba-lab/prepare_cavc_kernel.py` 把 `controller_v3/*.py`、`paired/run_pfba.py`、`decoupled/cpu_embedding.py` 嵌入 entry，保持目录布局：

```text
/tmp/pfba_cavc/
├── week/
│   ├── controller_v3/*.py
│   └── cpu_embedding.py
└── paired/
    └── run_pfba.py
```

这样 `controller_v3` 里的 `parents[2]/paired` 仍能解析到 `run_pfba`。20/20 嵌入文件哈希与仓库一致；entry 可编译；未 push。

---

## 2. 架构与建模实现

### 2.1 在线决策

每个源词位置：

1. 构造 34 维在线状态（已听原文、已提交译文、历史、运行时/成本、可选置信度 + mask）。
2. 编码状态 `z`。
3. READ 头给 `Q_READ`；WRITE 头对每个 `(B,H)` 给 `Q_WRITE`、error risk、no-emit risk、expected emit。
4. 统一打分：

```text
score(a) = Q(a) - lambda_T * C(a)
```

或开启 advantage baseline 时：

```text
A(a) = Q(a) - Q((4,6))
score(a) = A(a) - lambda_T * (C(a) - C((4,6)))
```

5. argmax over READ + `cfg.actions`。
6. WRITE 走 `ControlledRuntime.execute`：per-candidate `seed+i` 采样、批量翻译、RALCP、投票统计。
7. 可选 post-controller：STOP / ADD_2 / WAIT；ADD_2 用缓存候选扩展并重新 RALCP；临时结果不写 `committed`。
8. 句末用真实全文翻译一次。

### 2.2 动作空间

默认：

```text
B ∈ {2, 4, 6}, H ∈ {4, 6, 10}
9 个写动作 + 1 个 READ = 10 个可选动作
```

扩展：

```text
B ∈ {2,4,6,8,10}, H ∈ {2,4,6,8,10}
25 个写动作 + READ
```

动作特征 6 维：

```text
(b, h, b*h, log b, log h, ceil(0.6*b)/b)
```

### 2.3 34 维在线特征

```text
src_words, src_chars, tgt_words, tgt_chars,
length_ratio, normalized_lag, words_since_commit, no_emit_streak,
last_B, last_H, last_vote_top1, last_vote_margin,
last_vote_entropy, last_commit_count, last_empty,
src_end_punct, tgt_end_punct, last_capital,
subordinate_marker, current_clause_len,
pred_cost_ref, mt_cost_ref, pred_queue_work,
mt_queue_work, time_since_commit, arrival_slack,
mt_next_max_prob, mt_next_entropy, mt_prefix_logprob, pred_next_entropy,
mt_next_max_prob_available, mt_next_entropy_available,
mt_prefix_logprob_available, pred_next_entropy_available
```

在线特征严格只用已听原文、已提交译文、历史、运行时；标签/训练路径才可读全文、参考译文和 oracle。

### 2.4 损失

```text
L = alpha_Q * Huber(Q, y)
  + alpha_rank * pairwise ranking (same state_key, |delta| > epsilon, 32 pairs/state)
  + alpha_error * BCE(error_risk, incorrect>0)
  + alpha_noemit * BCE(no_emit, no_ralscp_output)
  + alpha_emit * Huber(softplus(emit_head), emitted)
```

`error_risk` / `no_emit` 是 raw logits，`expected_emit` 只在头部做一次 softplus。

### 2.5 成本模型

- `MeasuredCostTable`：`(predictor_id, translator_id, action) -> median(pred_s, mt_s)`，READ 零成本，回退 nearest -> global median -> 1e-3。
- `PowerLawCostModel`：`pred = a*b^r_b*h^r_h`、`mt = c*b^s_b*h^s_h`，对数线性拟合，指数夹在 [0,1]。
- `PipelineCostModel.marginal_wall`：预测/翻译两设备事件仿真，返回增量墙钟，不是 `T_P+T_M`。
- **注意**：生产路径目前用 `MeasuredCostTable.wall_s`，`PipelineCostModel` 还没有接进决策。

### 2.6 Post-controller

```text
STOP  : 提交当前 RALCP 结果
ADD_2 : 用缓存候选追加 2 条，重新 RALCP 全池
WAIT  : 丢弃临时结果，不提交；计算时间仍计入
```

阈值为手写启发式，未训练；`PostController` 目前没有在 `kaggle.py`/`cli.py` 里构造，属于“模块存在但未接入生产”。

---

## 3. 数据集与 Kaggle 计划

### 3.1 数据

| Split | 句数 | 源词数 | 平均词数 |
|---|---:|---:|---:|
| dev (newstest2013) | 3003 | 54,865 | 18.27 |
| test (newstest2015) | 2169 | 38,160 | 17.59 |

哈希桶（`sha256("13|" + id)` 前 8 字节 mod 10）：

```text
dev  buckets: 0=286, 1=295, 2=319, 3=292, 4=315,
              5=297, 6=300, 7=282, 8=315, 9=302
test buckets: 0=223, 1=226, 2=233, 3=244, 4=232,
              5=205, 6=206, 7=192, 8=226, 9=182
```

### 3.2 推荐切分

```text
训练：dev buckets 2–9 = 2422 句
调参：dev bucket 0   = 286 句
留出：dev bucket 1   = 295 句
7B 切片：留出前 64 句
干净确认：test:0140 之后（0012–0139 已被之前实验消耗）
```

注意：bucket 0/1 在 fold1/fold2/confidence 中用过；对 V3 是新模型，但不是历史干净留出。最终对外结论用 `test:0140+`。

### 3.3 预计时长

| 阶段 | 数据 | 预计 |
|---|---|---:|
| 本地 CPU 测试 | 无 | ~1 分钟 |
| Kaggle preflight | 8 句，3 states，2 rollouts | 0.5–1 h |
| dev 可行性 | 200–300 句 | 1.5–2.5 h |
| 全量 dev + 7B 64 | 2422 训练 + 295 留出 + 64 个 7B | 4–6 h |
| test:0140+ 确认 | 64–128 句 | 1–2 h |
| **合计** | | **7–11 h** |

额度：旧笔记 30 h/周、2026-11-14 00:00 UTC 刷新；今天已过刷新点，去 Kaggle 页面确认。单场约 12 h 上限，用 `run_dev` 的 deadline + resume。

### 3.4 上 Kaggle 的顺序

```bash
# 1. 先本地
pytest -q pfba-lab/week/controller_v3/tests

# 2. push kernel（需要你确认；会出现在你的账号下）
kaggle kernels push -p pfba-lab/kaggle-runs/pfba-cavc-v3

# 3. 拉结果
kaggle kernels output eunectesqaq/pfba-cavc-v3-preflight -p ./cavc_out
```

Kernel 默认跑：

```bash
python -m controller_v3.kaggle --mode preflight --limit 8 \
  --output-dir /kaggle/working/cavc
```

preflight 只验证：真实适配器能否加载、小/大 pair id、cost records、label -> targets -> train -> checkpoint -> rollout、产物写出、BLEU/COMET 状态。**不要拿它的 checkpoint 做质量结论。**

---

## 4. 最终代码审核

**结论：Ready for Kaggle preflight with fixes。0 Critical，6 Important，6 Minor。**

### 4.1 Important

#### I1. 训练状态的 History 全零，rollout 用实时 History（train/serve 偏斜）

- `kaggle.py:957-972`（`_targets_from_states`）与 `kaggle.py:1020-1055`（`_label_few_states`）调用 `builder.build(observed, committed, ...)` 时用默认 `History()`。
- `rollout.py:432-442` 在推理时传实时 `History` 和 `RuntimeInfo(time_since_commit=...)`。
- 后果：`words_since_commit`、`no_emit_streak`、`last_B/H`、`last_vote_*`、`last_commit_count`、`last_empty`、`time_since_commit` 等约 8/34 个特征在训练时恒为 0，在 rollout 时是分布外的真实值。价值函数被拟合到与评估不同的输入分布，学不到历史条件行为。
- 修法：从标注轨迹逐步重建 `History`；或明确把这些特征在训练/推理都置零并记录 ruling。

#### I2. 标注轨迹从不选 READ

- `kaggle.py:799-831`（`_best_write`）只在写动作里选最优；`kaggle.py:890-894`、`1029-1033` 即使 READ 的即时收益更高也提交一个写动作。
- `week_run.label_split`（`week_run.py:559-563`）允许 READ 赢；计划里的“READ leaves committed unchanged”在这里是空话。
- 后果：所有写都劣于 READ 时仍会提交最差的写，前缀和后续 successor 全部漂移，value targets 被污染。
- 修法：对 `(READ, *outcomes)` 取 argmax，保留 `b*h` tie-break；加 READ-wins 测试。

#### I3. 所有反事实动作共用一个 `next_state`

- `kaggle.py:980-990` 把一个 `next_state` 传给 `build_action_targets`；`targets.py:125-137` 把它赋给每个 outcome。
- `ActionOutcome.next_committed` 每个动作不同，但没有转成 per-action successor。
- 后果：Bellman backup 对所有动作相同，Q 差异几乎只来自即时奖励；READ 的 successor 尤其错（它应该看到下一个源词）。
- 修法：在 `labels.jsonl` 保留每个 outcome 的 `next_committed`/emitted words；为每个动作构造 successor；READ 用下一个 `observed`；或对非选中动作关闭 bootstrap。

#### I4. `trained=True` 在零 epoch 时也会写

- `kaggle.py:1171-1173` 在 `trainer.fit` 后无条件 `trained=True`；`trainer.py:238-249` 在 `cfg.epochs <= 0` 或 targets 为空时直接返回、不做优化。
- 后果：可以写出一个 `extra["trained"]=True` 但实际没训练的 checkpoint，`run_confirm` 会接受。
- 修法：`trained = training["epochs"] > 0 and training["n_samples"] > 0`。

#### I5. `compute_targets` 每个动作重复编码 next state（Kaggle 墙钟风险）

- `trainer.py:164-191` 对每个 `ActionTarget` 调 `_encode_states(target_network, [state])` + `write_values`。
- dev 约 2400 句 × 约 7 个 kept state × 10 个动作 ≈ 17 万 targets/epoch；同一个 next_state 被编码约 10 次；20 epochs = 数百万次 batch-1 GRU 前向。
- 后果：README 的 4–6 h dev 估计偏乐观。
- 修法：按 unique next_state + action tuple 缓存 one-step bonus；把 unique next states 批量编码；把 target 计算时间写进 dev summary。

#### I6. `run_confirm` 可能用大 pair 配小 pair 的 cost records

- `kaggle.py:1824-1841` 从 checkpoint 载入 `MeasuredCostTable`，但不校验记录的 `(predictor_id, translator_id)` 是否与当前解析的 pid/tid 一致。
- `MeasuredCostTable.predict` 找不到时会回退 global median，即小 pair 的 fallback。
- checkpoint 也没有把 adapter pair 写进 `extra`。
- 修法：checkpoint extra 存 model_pair；`run_confirm` 校验；不匹配时 raise；或加 pair_strict 模式。

### 4.2 Minor

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| M1 | 真实路径没有 `StageTimer(sync)`；calibration/rollout 用 `perf_counter` 无 device sync | `kaggle.py` 校准段；`runtime.py:166-172`；`run_pfba.py:687-693` | 用 resolved device 构造 sync closure，传 `StageTimer` |
| M2 | preflight 用全部 dev（含 holdout bucket 1）建词表 | `kaggle.py:1382,1447` | 用 train sentences |
| M3 | deadline 停止后覆盖已有 `summary.json`/`traces.jsonl` | `kaggle.py:1228-1242` | 新文件名或空 traces 时不覆盖 |
| M4 | `run_confirm` 在 checkpoint 缺 `extra["vocab"]` 时用 test 词表回退 | `kaggle.py:1787-1822` | 拒绝无词表 checkpoint，或要求显式 builder |
| M5 | README 说 test 0012–0139，代码是 0000–0139；resume 写成 `checkpoint.py` | `README.md:465-466,458` | 改成 `0000–0139`、`checkpoint.pt`/`labels.jsonl` |
| M6 | 没有测试保证提交的 kernel bundle 不过期 | `tests/test_kaggle_wiring.py:574-620` | 加 bundle 字节/哈希一致性测试 |

### 4.3 审核明确“不判”的项

- labeler batched `num_return_sequences` vs runtime per-candidate：计划已规定，只是采样流不同，不是泄漏。
- terminal reward 用词长差代理、不是 COMET：已文档化。
- ranking 只对写动作、空 state_key 跳过：计划 ruling。
- `PipelineCostModel` 未接进 rollout：已文档化的简化。
- post-controller 阈值手写：计划说明未调。
- `torch.manual_seed` 全局重置、`_encode_states` 固定 CPU/float32：已 deferred，无正确性影响。
- `save_checkpoint` 接受任意 `extra` 而 load 拒绝：故意用于 unsafe-pickle 测试。
- `_repair_jsonl_tail` 丢掉缺换行的完整最后一行：deferred，重标不损坏。
- cost table 忽略词数、labeler 不 shuffle：计划契约。

---

## 5. 建模一致性审核

**结论：Aligned with documented gaps；preflight 可跑，但先修 shared successor。**

### 5.1 对齐的元素

| Astra 元素 | 实现 | 状态 |
|---|---|---|
| Action-value 而非 one-hot | `CAVCNetwork` + `CAVCTrainer` Huber/Bellman | 实现 |
| READ/WRITE 统一价值尺度 | 共享 `z`、标量 READ 头、标量 WRITE 值头 | 实现 |
| READ 进 Bellman 候选集 | `targets.py:98-99`、`trainer.py:174-177` | 实现 |
| WRITE 动作条件评分 `(B,H)` | 6 维动作编码器 + 共享 trunk | 实现 |
| quality reward `w_m*matched - w_e*incorrect` | `reward.py:44-46` | 实现 |
| 显式 compute cost + `lambda_T` | 训练 backup 与选择时扣 | 实现 |
| one-step Bellman + frozen target | `trainer.py:149-191` | 实现 |
| 多任务损失 | Huber/rank/BCE/BCE/Huber | 实现 |
| 反事实一步标签 | `labeler.py` | 实现 |
| 嵌套候选 `seed+i` | `runtime.py`、`labeler.py` | 实现 |
| 严格在线/离线边界 | `features.py`、`labeler.py` | 实现 |
| post-controller 模块 | `post_controller.py`、`runtime.extend` | 实现但未接生产 |
| 测量/流水线成本模型 | `cost_model.py` | 部分（生产用串行表） |
| 大 pair id 进网络与 cost | `rollout.py`、`kaggle.py` | 实现 |
| 句子切分 / test:0140 保护 | `kaggle.py` | 实现 |
| 指标诚实 | `evaluator.py`、`cli.py` | 实现 |

### 5.2 结构性缺陷

**shared `next_state`**：`targets.py:134` / `kaggle.py:965-985` 给所有动作同一个 successor。方程 `Q(s,a)=r+gamma*V(s')` 机械上成立，但 backup 与动作无关，Q 退化成“一步 utility + 值形状的头”。在修复前，不能把闭环结果当作 action-value learning 的验证。

### 5.3 其他建模缺口

1. **READ 延迟惩罚扣在所有动作上**：`reward.py:62-72`、`targets.py:105-111`。按 Astra 应只扣 READ，或按动作后 backlog 计算。当前它是状态常数，在 argmax/ranking 中抵消。
2. **risk heads 未消费**：`error_risk`/`no_emit` 训练了，但 `PostController` 没在 `kaggle.py`/`cli.py` 构造；推理路径不用。
3. **生产用串行 cost**：`trainer.py:215-229`、`rollout.py:156-169` 用 `wall_s=pred+mt`；`PipelineCostModel.marginal_wall` 存在但没接。
4. **13/34 特征恒定**：5 个 runtime cost/queue/slack + 4 confidence + 4 mask 在 rollout 中恒 0。
5. **没有 fixed-(4,6) baseline** 走同一 per-candidate 路径；README 只是文字 caveat。
6. **labeler batched generate vs runtime per-candidate**：两条流不同。
7. **没有 Brier/ECE 校准**；`error_risk`/`no_emit` 的 sigmoid 不能当概率。
8. **没有 terminal/EOS target**；`terminal_reward` 是 dead code。
9. **advantage baseline 是常数平移**：`A=Q-Q(a0)` 不改变 argmax，也不进训练；它是 baseline-shifted scoring，不是 advantage learning。
10. **Kaggle summary 没有 sentence-paired 区间**；`paired_bootstrap` 存在但没在 `kaggle.py` 调用。

### 5.4 建模审核的最终判断

- **Modeling consistency**：Aligned with documented gaps。
- **Ready for Kaggle preflight**：With fixes。
- **Top 3 首次运行风险**：
  1. action-independent Bellman + write-only 轨迹 + stride-4 多步 successor，最容易复现旧的“离线收益升、闭环 BLEU 降”。
  2. 真实模型和大 pair 接线未验证；`run_confirm` 缺 pair guard；labeler/runtime 采样流不同。
  3. cost/risk/post-controller 不在决策路径，跑出来的不是完整 Astra 控制器。

---

## 6. 逐 Task 审核与修复记录

| Task | 内容 | 审核结论 | 修掉的关键问题 |
|---|---|---|---|
| 1 | action/config/features/text encoder | Approved | 从句标记不能含冠词；History 字段映射 |
| 2 | CAVC network + trainer | Approved | emit 只 softplus 一次；error/noemit raw logits；ranking 仅写动作；bootstrap 默认含 READ |
| 3 | cost model + runtime | Approved | vote 分词必须与 RALCP 一致（`split(" ")`）；READ 不进 measured table；extend 0 分支 |
| 4 | reward + labeler | Approved | 短/空候选池必须 raise；oracle 时间与 action mt 分离；文档修正 |
| 5 | rollout + post controller + evaluator | Approved | `network.eval()` 缺失（dropout 活跃）；空源 EOS；ADD_2 累计计时；WAIT 完成时间 |
| 6 | CLI/checkpoint/targets/README | Approved | `weights_only=True`；smoke BLEU 强制 unavailable；格式版本；CLI smoke 测试 |
| 7 | Kaggle 接线/校准/打包 | Approved | 大 pair id 未注入 rollout；deadline 不约束校准/训练/rollout；dev 词表混入 holdout；未训练 checkpoint 被 confirm 当已训练 |

每个 Task 的 Critical/Important 都经过 fix round + scoped re-review，最终全部 Approved。

---

## 7. 冻结决定与下一步

### 7.1 为什么冻结在 5:30

- 代码、测试、逐 Task 审核、两份最终审核、Kaggle 打包都已完成。
- 6 个 Important 里至少 3 个（History 偏斜、READ 不被选、shared successor）会直接偏置学习目标，修完约 1.5–2 小时。
- 5:30 硬塞会把分支留在半修状态，反而不能跑。
- 所以冻结 `4984b63`，作为 **preflight-plumbing 版本**。

### 7.2 下一步修复顺序（建议一次性做完再复审）

1. **READ selection**：`_best_write` -> `argmax(READ, *outcomes)`，保留 `b*h` tie-break；加 READ-wins 测试。
2. **History propagation**：标注时逐步构造 `History`；训练/推理同一分布；加 train/serve 一致性测试。
3. **Per-action successor**：`labels.jsonl` 保留 `next_committed`；`build_action_targets` 接受 per-row successor；READ 用下一个 observed；加测试。
4. **zero-epoch trained guard**：`trained = epochs>0 and n_samples>0`。
5. **compute_targets 缓存**：按 unique next_state + actions 缓存/批量编码；把 target 时间写进 summary。
6. **confirm pair guard**：checkpoint extra 存 model_pair；`run_confirm` 校验；不匹配 raise。
7. **Minor**：StageTimer(sync)、preflight train-only vocab、artifact 不覆盖、test-vocab 拒绝、README 修正、bundle-staleness 测试。
8. 修完后跑 preflight（0.5–1 h）-> 200–300 句 dev（1.5–2.5 h）-> 全量 dev + 7B（4–6 h）-> test:0140+（1–2 h）。

### 7.3 不要做的事

- 不要拿当前 checkpoint 的 BLEU/LAAL 当结果。
- 不要在 `test:0140+` 上调任何东西。
- 不要把 per-candidate controller 和 batched `num_return_sequences` baseline 直接比较。
- 不要把 oracle-prefix match 或 virtual replay time 当成 BLEU/COMET/LAAL。
- 不要在 preflight 的 8 句上做模型选择。

---

## 8. 文件与命令速查

### 8.1 代码

```text
pfba-lab/week/controller_v3/
pfba-lab/prepare_cavc_kernel.py
pfba-lab/kaggle-runs/pfba-cavc-v3/
```

### 8.2 审核记录

```text
.superpowers/sdd/2026-11-15-cavc-controller-v3/progress.md
.superpowers/sdd/2026-11-15-cavc-controller-v3/final-review-summary.md
.superpowers/sdd/2026-11-15-cavc-controller-v3/task-1-report.md ... task-7-report.md
.superpowers/sdd/2026-11-15-cavc-controller-v3/kaggle-analysis.md
.superpowers/sdd/2026-11-15-cavc-controller-v3/astra-design-summary.md
```

### 8.3 常用命令

```bash
# 测试
"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests

# CPU smoke
cd pfba-lab/week
"D:/MINICONDA/envs/py312/python.exe" -m controller_v3.cli --smoke --output-dir <tmp>

# 重新生成 Kaggle kernel
"D:/MINICONDA/envs/py312/python.exe" pfba-lab/prepare_cavc_kernel.py

# push（需要确认）
kaggle kernels push -p pfba-lab/kaggle-runs/pfba-cavc-v3

# 拉结果
kaggle kernels output eunectesqaq/pfba-cavc-v3-preflight -p ./cavc_out
```

### 8.4 关键参数

```text
seed = 13
min_start = 3
arrival_gap = 0.3 s
ralcp_threshold = 0.6
default actions = B{2,4,6} x H{4,6,10}
advantage baseline = (4,6)
gamma = 0.99
lambda_l = 0.05
lambda_t = 0.10
weight_m = 1.0
weight_e = 2.0
```

---

## 9. 术语

| 术语 | 含义 |
|---|---|
| TAF | Translation by Anticipating Future，LLM 预测未来原文再翻译 |
| RALCP | 按词前缀投票，支持数达到候选数比例才提交 |
| LAAL | 内容延迟，以源词计 |
| B / H | 未来条数 / 每条最大 token 数 |
| READ | 继续听，不写 |
| oracle | 模型自己用全源文翻译的续写，只用于打标签 |
| CAVC | Cost-Aware Value Controller，本项目的新控制器 |
| shared successor | 所有动作共用同一个 `next_state` 的缺陷 |
| per-candidate seed | 候选 `i` 用 `seed+i`，保证 B=2 是 B=4 的前两条 |

---

*报告结束。详细逐条审核在 `.superpowers/sdd/2026-11-15-cavc-controller-v3/`。*
