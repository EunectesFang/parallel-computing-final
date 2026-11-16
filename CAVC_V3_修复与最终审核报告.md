# CAVC V3 修复波与最终审核报告

**日期**：2026-11-15  
**仓库**：`C:\Users\Fang\Desktop\并行计算\期末项目说明\翻译`  
**分支**：`cavc-controller-v3`  
**代码 HEAD**：`7124a20`（修复波）；计划文档 `0855bd5`  
**状态**：修复波 + final fix wave 完成并复审；255 个 controller 测试 / 261 个 `pfba-lab/week` 测试通过；`8c271f4` 上 preflight 已提交 Kaggle（运行中）。

> 这是第一份报告《CAVC_V3_实施与审核报告.md》之后的新版本。第一份记录的是修复前的工程实现与问题；本份记录修复波（Task 8–10）实际改了什么、验证了什么、还剩什么。

---

## 一页结论

| 项 | 结果 |
|---|---|
| 修复波 | Task 8（建模核心）、Task 9（真实运行路径）、Task 10（严格性/产物/打包）全部实现并逐 Task 审核通过 |
| 测试 | 244 controller tests；250 `pfba-lab/week` tests；`compileall`、`git diff --check` 通过 |
| 三个结构性问题 | per-action successor、训练 History、READ 轨迹均已在代码和测试中修复 |
| 其他修复 | terminal 前缀、stride=1、marginal 时钟、risk 校准门控、post-controller 接线、fixed-(4,6) 同路径基线、candidate parity、feature diagnostics、pair/vocab 严格校验 |
| Kernel | 重新生成并 byte-verify，`prepare_cavc_kernel.py --check` 返回 `bundle up to date` |
| 真实模型实验 | 尚未运行；没有 BLEU/COMET/LAAL 结果 |
| Kaggle 估计 | 30 GPU h 预算下重规划（§5.4）：preflight 1 h + 200 句门控 3.5 h + 600 句主实验 6 h + large-pair preflight 1 h + 确认 2 h，合计 ≤14 h |
| 最终复核 | 建模 re-review + 整波 code review 已完成；final fix scoped re-review 结论 findings addressed |

**一句话**：你指出的三个“会破坏 Action-Value 含义”的问题已经真正修掉；现在代码可以进入 preflight 和 200–600 句 dev 可行性，但仍不能拿 CPU smoke 或未训练 checkpoint 当质量结论。

---

## 1. 交付物与验证

### 1.1 提交

```text
code HEAD: 7124a20
docs HEAD: 0855bd5
branch:    cavc-controller-v3
base:      7aca8ec (main)
```

修复波提交：

```text
44010bd  Fix CAVC modeling core: per-action successors, history, READ trajectory, terminal
27db117  Fix CAVC review round 1: terminal prefix, shared clock, early label stop
e66ca12  Wire runtime cost, risk/post controller, fixed baseline, feature diagnostics
1a1094e  Fix CAVC Task 9 review: marginal clocks, trainer cost, calibration gate, confirm vocab
f97fab1  Harden CAVC experiment artifacts, timing, and kernel bundle
7124a20  Fix CAVC deadline artifact pointer and calibration provenance
```

### 1.2 验证

```bash
"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week/controller_v3/tests
# 244 passed

"D:/MINICONDA/envs/py312/python.exe" -m pytest -q pfba-lab/week
# 250 passed

"D:/MINICONDA/envs/py312/python.exe" -m compileall -q pfba-lab/week/controller_v3
# OK

"D:/MINICONDA/envs/py312/python.exe" pfba-lab/prepare_cavc_kernel.py --check
# bundle up to date
```

`kaggle_entry.py` SHA-256：`736f5716c21ae66f163b7ae84441a118b729e11cf69a40dbb149b39b9dcf5e92`。

---

## 2. 三个结构性问题：修复证据

### 2.1 Per-action successor：Bellman backup 现在是 action-conditioned

**原来的问题**：`build_action_targets` 给每个动作同一个 `next_state`；`compute_targets` 对所有动作共用一个 bootstrap，Q 差异几乎只来自即时奖励。

**现在的实现**：

- `kaggle.label_sentence` 为每个动作构造独立 successor：
  - READ：`observed_next = words[:read_pos+1]`，`committed` 不变，`tracker.on_read()`。
  - WRITE：`observed_next = words[:read_pos+1]`，`committed = outcome.next_committed`，`tracker.on_write(outcome)`。
- 每个 successor 有自己的 `History`；标签 JSONL 持久化 `next_committed`、`emitted`、vote stats、successor 输入。
- `targets.build_action_targets` 用 `next_states[i]` 对应 `outcomes[i]`；长度不匹配直接 raise。
- `trainer.compute_targets` 用每个 target 自己的 `next_state`；按 unique successor 批量编码并缓存 one-step bonus。
- `cli._run_smoke` 和 `test_integration.make_dataset` 也走 per-action successor，不再走 legacy shared-successor 路径。

**测试**：

- 不同 WRITE 动作产生不同 successor。
- 两个动作的 `compute_targets` 目标不同。
- 缓存命中/批量编码有 call-count 测试。
- `tests/test_modeling_core.py`、`test_integration.py` 覆盖。

### 2.2 训练 History：训练与 rollout 使用同一套状态更新

**原来的问题**：训练状态用默认 `History()`（全零），rollout 用实时 History；约 8/34 个特征 train/serve 偏斜。

**现在的实现**：

- `features.HistoryTracker` 是唯一的状态更新路径：
  - `on_read()`：`words_since_commit+1`、`no_emit_streak+1`、`last_empty=True`、`last_commit_count=0`、`last_B/H=0`。
  - `on_write(outcome)`：写 `last_B/H`、vote 统计、`last_commit_count=emitted`；提交成功则清零滞后，否则递增。
  - `build(...)` 与 `snapshot()` 供标注和 rollout 共用。
- `rollout.run_rollout` 和 `kaggle.label_sentence` 都使用它；没有手写 `History(...)`。
- 标签 JSONL 存 pre-action tracker snapshot；`_targets_from_rows` 重放出同一个 transition。
- `features.advance_virtual_clock` 是标注和 rollout 共用的时钟递推；`time_since_commit` 在两边都由同一递推和 marginal 时间推进。
- `feature_diagnostics` 输出每维 mean/std/nonzero/missing 与 train-serve 偏移，并标记“训练恒定、推理变化”的特征；`feature_stats.json` 写入 preflight/dev。

**测试**：

- 离线 History 重放等于在线构建。
- 同一动作序列的标注时钟与 rollout 时钟一致。
- 常量特征检测。
- `tests/test_modeling_core.py`、`test_features.py`。

### 2.3 READ 轨迹：READ 真实进入训练，且目标不是即时奖励 argmax

**原来的问题**：`_best_write` 只在写动作里选，标注轨迹从不选 READ；READ 的延迟惩罚还是状态常数。

**现在的实现**：

- 行为策略每个决策点：`read_prob=0.25` 选 READ，否则固定 `(4,6)`；用来覆盖状态，不用来决定 Bellman 目标。
- 每个 kept position 都标注全部 `cfg.actions` + READ；READ 有独立 outcome 和 successor。
- READ 的 successor 保持 committed、推进 observed；WRITE 的 successor 用 `next_committed`。
- READ 可以在网络 Q 更高时被选中（`test_read_wins`）。
- 延迟惩罚改为动作后的 backlog：
  `backlog_after = max(0, src_words_after - (current_tgt + emitted)/rho)`。
  成功提交可以降低 backlog；no-emit write 与 READ 的 compute 成本不同。
- 句末 terminal：`terminal_value` 用“已提交前缀 + oracle 续写”与参考的词长差代理；没有 bootstrap；不是 BLEU/COMET。
- `stride=1` 是真正的一步转移；`stride>1` 直接 raise，除非实现真正的多步累计折扣目标（本波未实现）。

**测试**：

- READ 保持 committed、推进 observed。
- READ 在 Q 更高时被选中。
- 延迟随 `emitted` 改变。
- terminal 无 bootstrap，且更长的正确前缀不会让 terminal 值更负。
- `stride>1` raise。
- `tests/test_modeling_core.py`。

---

## 3. Task 9/10 的其他修复

| 修复 | 实现 | 验证 |
|---|---|---|
| Pipeline 边际墙钟 | `ControlledRuntime` 跟踪 `t_pred_free`/`t_mt_free`，`StepResult.marginal_wall_s`；calibration 记录第六字段 | 手算调度测试；rollout 用 marginal 计费 |
| 训练/推理成本一致 | 标注、successor 时钟、trainer Bellman cost 都优先 marginal；serial 仅作有文档的 fallback | 标注 vs fixed rollout 时钟一致测试 |
| Post-controller 接线 | `--post-controller` 构造 `PostController`，传 selected `no_emit`、vote stats、`spent_s`；关闭时记录 `disabled` | on/off summary 测试 |
| Risk 校准 | Brier、ECE bucket error、no-emit 校准；`risk_calibration` 只在可证明训练过的 checkpoint 上运行 | 手算 Brier/ECE；未训练 -> `unavailable` |
| Fixed `(4,6)` 基线 | `run_fixed_rollout` 用同一 runtime/采样/翻译/RALCP/计时，永远 `(4,6)`，不 READ | no-READ 测试；`paired.json` |
| Feature diagnostics | 34 维 mean/std/nonzero/missing、train-serve 偏移 | 常量特征检测测试 |
| Candidate parity | `candidate_mode=per_candidate` 默认，`seed+i` 与 runtime 一致；`batched` 仅 ablation | parity 测试 |
| Checkpoint 严格性 | `extra` 存 `model_pair`、`vocab`、`feature_stats`；confirm 校验；不匹配 raise | pair/vocab/stats mismatch 测试 |
| Confirm 无 test 词表 | 必须 checkpoint 或显式 frozen builder+network；否则 raise；`vocab_source="supplied_frozen"` | refusal 测试 |
| `trained` 诚实 | 只有实际优化 epoch 且样本数 > 0 才 `trained=True`；零 epoch 不写 checkpoint | zero-epoch/deadline 测试 |
| Deadline 产物 | 不覆盖非空 summary/traces；`deadline_artifacts` 在序列化前写入；早期返回 schema 有 null 键 | deadline artifact 测试 |
| Bundle | `prepare_cavc_kernel.py --check`，byte-match 测试 | `bundle up to date` |
| LABEL_FORMAT | 5；旧格式与 mixed candidate mode 拒绝 | stale-format 测试 |

---

## 4. 仍然存在的风险与建模假设

这些不影响“能跑 preflight/feasibility”，但在写论文/报告时要明确：

1. **多步 target 未实现**：`stride>1` 直接 raise。当前只支持真正的一步 Bellman。代价是标注量约 4×。
2. **Terminal 仍是长度差代理**：`terminal_reward` 是词长差（加可选延迟），不是 BLEU/COMET；它是 action-independent 的句末值。
3. **Trainer 的 Bellman cost 是冷调度估计**：trainer 没有每步设备状态，`PipelineCostModel.marginal_wall(0,0,0,0)` 在无设备状态时数值上等于 serial；真正 measured marginal 在标注/rollout 路径里。这是一个已文档化的近似。
4. **Confirm 的 risk calibration**：没有 held-out `ActionTarget` 时为 `unavailable`；confirm 不重新校准，避免在 test 上调参/校准。
5. **DAgger 未实现**：行为策略只有 READ 0.25 + 固定 `(4,6)`；没有按当前网络策略做多轮重标注。状态覆盖仍有偏。
6. **Text encoder ablation 未跑**：文本分支存在，但还没有证明它有用。
7. **Model-pair quality labels**：训练标签只来自小模型对；大模型对只做 preflight/rollout/成本校准。跨 pair 迁移未验证。
8. **Post-controller 阈值**：手写启发式，未训练；默认关闭，除非显式 `--post-controller`。
9. **13/34 特征可能仍恒定**：runtime cost/queue/slack 和 confidence 在默认 rollout 中可能为 0；`feature_stats.json` 会暴露这一点。
10. **7B/Tower 真实路径未验证**：只做了 fake 测试；必须先用 4–8 句 large-pair preflight。
11. **没有真实 BLEU/COMET/LAAL**：CPU smoke 的 BLEU `unavailable`、COMET `skipped` 不是结果。
12. **Per-candidate 采样可能更慢**：B 次单候选 `generate` 不是一次 batched；成本模型/日志会量出来，但不能假设它和旧 TAF 同速。

---

## 5. 数据集与 Kaggle 计划（更新）

### 5.1 数据

| Split | 句数 | 源词数 |
|---|---:|---:|
| dev (newstest2013) | 3003 | 54,865 |
| test (newstest2015) | 2169 | 38,160 |

推荐：

```text
训练：dev buckets 2–9 = 2422
调参：dev bucket 0   = 286
留出：dev bucket 1   = 295
7B 切片：留出前 64
干净确认：test:0140+（test:0000–0139 已消耗）
```

### 5.2 时长（stride=1 后）

| 阶段 | 数据 | 预计 |
|---|---|---:|
| 本地 CPU 测试 | 无 | ~1 分钟 |
| 小 pair preflight | 8 句，3 states，2 rollouts | 0.5–1 h |
| 大 pair preflight | 4–8 句 | 0.5–1 h |
| 200 句可行性 | 200 句，stride 1 | 2–3.5 h |
| 300 句可行性 | 300 句 | 3–5 h |
| 600 句 dev | 600 句 | 5–9 h |
| 全量 dev + 7B 64 | 2422 + 295 + 64 | **12–19 h**，需跨 session |
| test:0140+ 确认 | 64–128 句 | 1–2 h |

**建议第一次 value-learning 实验用 200–600 句，不要直接上全量。** 全量要分两场 Kaggle session，用 `run_dev` 的 `max_hours`、per-sentence labels resume、kept checkpoint。

### 5.3 上 Kaggle 的顺序

```bash
# 1. 本地
pytest -q pfba-lab/week/controller_v3/tests

# 2. 检查 kernel
python pfba-lab/prepare_cavc_kernel.py --check

# 3. push（需要你确认；会出现在你的账号下）
kaggle kernels push -p pfba-lab/kaggle-runs/pfba-cavc-v3

# 4. 拉结果
kaggle kernels output eunectesqaq/pfba-cavc-v3-preflight -p ./cavc_out
```

preflight 只验证接线/产物，不做质量结论。

### 5.4 GPU 30 小时预算下的阶段 3/4 重规划（2026-11-16）

总配额按 **30 GPU h** 计。原计划“全量 dev + 7B 64”需要 12–19 h，加上前面阶段会吃掉 18–25 h，只剩 5–12 h 给确认和重跑，风险过高；且 7B 64 句跨 pair 迁移尚未做 large-pair preflight，属于未验证的昂贵实验。因此：

| 阶段 | 内容 | 预计 | 硬上限 |
|---|---|---|---:|
| 1 | 小 pair preflight（已 push，运行中） | 0.5–1 h | 1.5 h |
| 2 | 200 句 dev 门控：labels → train → dev holdout rollout + fixed `(4,6)` paired | 2–3.5 h | 3.5 h |
| 3 | 600 句主实验：resume 复用 200 句 labels，补 400 句，重训 600 句并 rollout | 3.5–6.5 h | 6 h |
| 3b | large-pair preflight（4–8 句，只验证 7B/Tower 接线） | 0.5–1 h | 1 h |
| 4 | `test:0140+` 干净确认（64 句，冻结 checkpoint，不调参） | 1–2 h | 2 h |
| — | **小计** | **7.5–14 h** | **14 h** |
| — | 预留：失败重跑、输出丢失、Kaggle 排队/重启 | — | **16 h** |

**阶段 3 的变化**：

1. 不做全量 2422 句；主实验改为 **600 句**。若阶段 2 的 200 句结果健康（`trained=True`、rank loss 非零、feature_stats 不退化、BLEU/LAAL 不劣于 fixed `(4,6)`、无 deadline/产物异常），才 resume 到 600；否则先修再花额度。
2. 不做 7B 64 句切片，只做 **4–8 句 large-pair preflight**。跨 pair 迁移留到后续周配额刷新后再跑。
3. resume 前先把 200 句的 `summary.json`/`traces.jsonl`/`checkpoint.pt` 复制成带规模后缀的文件；`run_dev` 重跑会重训并覆盖同名 summary，labels 才是真正的续跑依据。

**阶段 4 的变化**：

- 只跑 **64 句**（不是 128），单独 kernel，通过私有 dataset 挂载 dev 产出的冻结 checkpoint；不重新训练、不校准、不看 test 调任何东西。
- 确认结果的比较对象是阶段 3 的 dev holdout 与同一 kernel 里的 fixed `(4,6)` 基线，不能把 dev 数字直接当 test 结论。

**可选拉伸（仅在阶段 3 实际 ≤6 h 且配额仍 ≥16 h 时）**：resume 到 1200 句，预计再 +6–9 h，总计 ≤23 h，仍留 ≥7 h 缓冲。不做全量 2422 的承诺。

**额度守则**：每个 dev run 设 `--max-hours`（200 句 3 h、600 句 8 h），到点写 deadline 产物并保留 checkpoint；每次跑完立刻 `kaggle kernels output` 下载；以 Kaggle 账号页的实际 GPU 剩余小时为准，CLI 没有可靠的 quota 查询。

### 5.5 阶段 1/2 实际结果（2026-11-16）

**Preflight**（5 分钟）：3 states / 30 targets / 20 epochs / 2 rollouts，`trained=True`、`rank_loss=0.687`、BLEU 17.96（2 句，仅验证接线）、COMET skipped，全部产物写出。

**200 句 dev 门控**（2 h 43 min，未触发 deadline）：

| 项 | 结果 |
|---|---|
| 标注 | 165 句训练句、164 行、2308 states、23080 targets（18460 训练 / 4620 风险校准） |
| 训练 | 20 epochs，`trained=True`，`rank_loss=0.668`，`target_wall_s=220.7` |
| 特征 | `feature_stats` 无 `constant_train_varies_serving` 警告 |
| 风险校准 | error ECE 0.024 / Brier 0.188；no_emit ECE 0.036 / Brier 0.214 |
| Controller（n=17） | BLEU **27.72**，LAAL **3.406**，completion 9.98 s，READ 42.2% |
| Fixed `(4,6)`（n=17） | BLEU 20.40，LAAL 2.376，completion 9.40 s |
| Paired bootstrap | BLEU **+9.12 [2.37, 16.47]**；LAAL **+1.03 [0.53, 1.58]**（变差）；completion +0.58 [-2.10, 3.40]（不显著） |

**判读**：接线、训练、产物、风险校准和 paired 基线全部正常，学习信号真实存在；但 controller 用 READ + 更大的动作（b6h10 69 次、b2h10 32 次）换质量，LAAL 显著变差。这是预期的质量-延迟权衡，不是“免费提升”。n=17 太小，不能当最终结论；600 句主实验（n≈60）才是主结果。

**600 句主实验**：因 `--limit` 从 200 改到 600 会重建 train-only 词表/rho，I5 的 fingerprint guard 会正确拒绝复用 200 句 labels，所以改为 **fresh 运行**（`--limit 600`，预计 8–9 h，`--max-hours 10`）。200 句产物快照在 `pfba-lab/kaggle-runs/results/cavc-dev-200/`。

> 2026-11-16 更新：600 句 kernel 启动约 6 分钟后按用户要求停止并删除（尚无结果产出）。生成好的 bundle 保留在 `pfba-lab/kaggle-runs/pfba-cavc-v3-dev600/`，需要时重新 push 即可。

---

## 6. 最终复核状态

### 6.1 建模 re-review（已完成）

- 三项结构性问题：**全部 Fixed**。
  - per-action successor：`kaggle.py:1336-1405`、`targets.py:133-145`、`trainer.py:184-218`；测试 `test_modeling_core.py:147-188,341-373`。
  - 训练 History：`features.py:301-410`、`kaggle.py:1440,1482-1584`、`rollout.py:529,710`；无手写 `History(...)`。
  - READ 轨迹：`read_prob=0.25`、`targets.py:143-145`、post-action backlog；测试 `test_modeling_core.py:212-245,553-603`。
- 不变量：action-conditioned backup、READ in backup、post-action delay、terminal no bootstrap、marginal clocks、candidate nesting、no leakage —— 全部成立。
- 结论：**Modeling consistency = Aligned with documented gaps**；**preflight = Yes**；**dev/quality claim = With fixes**。

### 6.2 dev 前必须修的两项

1. **多余的 full-source oracle 翻译**：`kaggle.py:1393` 重算 `oracle_words`，只为持久化一个 `_targets_from_rows` 不读的计数；stride-1 下约 2–3× 最贵调用。删掉。
2. **terminal target 仍 action-independent**：`kaggle.py:1408-1417` 用 behavior action 的 successor 算一个 `terminal_value`，`targets.py:162-165` 赋给所有动作。改成每个 outcome 用 `next_committed` + oracle 续写算自己的 terminal value。

### 6.3 剩余风险（按优先级）

1. terminal 的 action-independence（上条）。
2. predicted-vs-measured marginal / trainer 冷调度成本：标注用预测 marginal，rollout 用实测 marginal，trainer 的 `C` 在零设备时间下退化为 serial。真实硬件上 `time_since_commit` 和 cost target 可能漂移。
3. post-controller 的 `no_emit >= 0.60` 是未校准 sigmoid 的硬阈值；默认关闭，若启用需标为启发式；标注不跑 post-controller，`ADD_2` 会使 `last_B`/vote stats 偏离训练分布。
4. DAgger / policy-state coverage 未实现；行为只有 READ 0.25 + fixed (4,6)。
5. 多步 stride 未实现（stride>1 raise）。
6. 7B/Tower 真实路径未验证。
7. 13/34 特征可能仍恒定；`feature_stats.json` 会暴露。

### 6.4 整波 code review（已完成）

- 结论：**preflight = Yes；dev = With fixes**。
- 验证：244 tests、bundle `--check` up to date、smoke BLEU unavailable/COMET skipped。
- **Critical：pairwise ranking loss 在生产 batch 下几乎不触发。** 全局 shuffle 后同一 `state_key` 的 10 行很少同 batch；模拟 48,000 states × 10 rows / batch 128，只有约 0.03% 的同状态 pair 存活；`rank_loss` 是 0 或可忽略。现有 fit 测试都在单 batch 上，所以没暴露。
- Important：
  1. 网络初始化未 seed；同一 labels + `cfg.seed` 每次权重不同，保存的 seed 不能复现。
  2. oracle 每状态重复翻译 2–3 次（`labeler` 一次、`kaggle.py:1393` 一次、terminal 一次），约多 25% MT 工作。
  3. terminal target 仍对同一句末状态的所有动作共用。
  4. `run_confirm` 接受无词表的 supplied builder，可能把空/UNK text ids 喂进训练过的网络。
  5. dev resume 未固定 label-generation context（pair ids、`read_prob`、vocab/rho fingerprint），换 pair 或改 `--limit` 可能静默复用旧标签。
- Minor：deadline artifact 指向空 traces、READ 重复行、未使用的 config 字段、重复 helper、单 adapter 注入被覆盖、confirm 不校验 caller cfg、plain-cost 在 label_sentence 不 warn、post-controller 未校准阈值措辞等。

### 6.5 最后一轮 fix（已完成并复审）

WIP `8c271f4` 已实现全部 final-review findings：

1. 按 whole state group 组 batch，恢复 ranking loss；多 batch pair-coverage 与 group-integrity 测试通过。
2. `_components` 在构建网络前 seed 初始化。
3. `CounterfactualLabeler.oracle_words` 按 `(source, committed)` 缓存；`label_sentence` 复用计数与 terminal READ 前缀。
4. per-action terminal value（每个动作的 `next_committed` + 自己的 oracle 续写），`LABEL_FORMAT` 6。
5. confirm 无词表 builder 拒绝；resume 标签 stamp pair/`read_prob`/`rho`/vocab fingerprint。
6. READ dedupe、confirm cfg 校验、plain-cost warn-once、README、`summary.deadlineN.json` 等 cheap minors。

scoped re-review 结论：**findings addressed；preflight = Yes；dev = Yes with documented gaps**。验证：255 controller / 261 `pfba-lab/week` tests、`compileall`、kernel `bundle up to date`、CPU smoke 正常。记录见 `.superpowers/sdd/2026-11-15-cavc-fix-wave/final-fix-review.md`。非阻塞残留：oracle 缓存不清理、warn-once 用 `id()`、per-action terminal 在句末增加少量 oracle 调用。

---

## 7. 不要做的事

- 不要拿 CPU smoke 的 `BLEU unavailable`、`COMET skipped` 当模型有效性。
- 不要拿未训练/零 epoch 的 checkpoint 做确认。
- 不要把 serial `pred+mt` 当流水线墙钟；用 `marginal_wall_s`。
- 不要在 `test:0140+` 上调任何东西。
- 不要直接拿 per-candidate controller 和 batched `num_return_sequences` baseline 比；baseline 必须走同一路径。
- 不要在 7B/Tower 未做 large-pair preflight 前跑 64 句切片。
- 不要把 terminal 长度差代理当 BLEU/COMET。

---

*报告结束。逐 Task 审核、修复报告和最终复核记录在 `.superpowers/sdd/2026-11-15-cavc-fix-wave/`。*
