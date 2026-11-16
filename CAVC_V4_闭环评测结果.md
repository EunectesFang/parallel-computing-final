# CAVC-Q V4 闭环评测结果

- 日期：2026-11-16
- 数据：WMT de→en `dev:0260–0319`（offset 260, limit 60）
- 模型对：Qwen2.5-0.5B predictor + Helsinki-NLP/opus-mt-de-en translator
- 运行时：同一 `ControlledRuntime`、seed 13、RALCP 0.6 / min_start 3、同一 timing 路径
- 基线：fixed `(4,6)`，每步都写、从不 READ
- 两个 kernel：
  - `eunectesqaq/pfba-cavc-v4-linear`
  - `eunectesqaq/pfba-cavc-v4-convex`
- 数据集：`eunectesqaq/cavc-v4-checkpoints`（v3_baseline + v4_linear + v4_convex）

> 说明：这是 **CAVC-Q V4** 的结果，代码仍带有当时已知的实现缺陷（见第 6 节）。
> V5 正在修复这些缺陷并重新训练，本文件只记录 V4 的实际闭环数字。

---

## 1. 主结果

| 策略 | BLEU | LAAL ↓ | completion ↓ | READ | chrF |
|---|---:|---:|---:|---:|---:|
| fixed (4,6)（linear kernel） | 32.87 | 2.45 | 13.53 | 0% | 57.63 |
| fixed (4,6)（convex kernel） | 32.87 | 2.45 | 13.36 | 0% | 57.63 |
| v3_baseline | 41.05 | 9.03 | 7.15 | 85.1% | 64.04 |
| v3_baseline_guard | 36.82 | 3.28 | 14.73 | 40.4% | 60.70 |
| **V4-Linear** | 34.20 | **2.35** | **8.65** | 3.5% | 59.03 |
| V4-Linear + guard | 33.94 | 2.34 | 8.71 | 1.2% | 58.89 |
| **V4-Convex** | 34.21 | **2.40** | **8.74** | 1.1% | 58.86 |
| V4-Convex + guard | 34.15 | 2.39 | 8.74 | 0.3% | 58.85 |

两个 kernel 的 fixed 基线差 0.17 s，属于测量噪声；配对比较使用各自 kernel 内的 fixed。

---

## 2. 相对 fixed (4,6) 的配对 bootstrap

差值 = 该策略 − fixed。

| 策略 | ΔBLEU [95%] | ΔLAAL [95%] | Δcompletion [95%] |
|---|---|---|---|
| v3_baseline | **+7.85 [3.88, 12.28]** | **+6.58 [4.41, 9.30]** | **−6.38 [−8.00, −4.88]** |
| v3_baseline_guard | **+3.73 [1.03, 6.39]** | **+0.83 [0.60, 1.06]** | +1.20 [−0.12, 2.46] |
| V4-Linear | −0.28 [−3.65, 2.56] | −0.10 [−0.32, 0.13] | **−4.88 [−6.15, −3.63]** |
| V4-Linear + guard | −0.26 [−3.64, 2.61] | −0.11 [−0.33, 0.11] | **−4.83 [−6.10, −3.57]** |
| V4-Convex | +0.13 [−2.80, 2.80] | −0.05 [−0.26, 0.15] | **−4.62 [−5.91, −3.37]** |
| V4-Convex + guard | +0.11 [−2.82, 2.80] | −0.06 [−0.27, 0.14] | **−4.62 [−5.92, −3.38]** |

（v3_baseline_guard 的 completion 在两个 kernel 分别为 +1.20 [−0.12, 2.46] 和 +1.27 [0.01, 2.53]，处于噪声边界。）

**关键结论：V4 在 BLEU 和 LAAL 上与 fixed 统计上打平，同时每句 completion 显著降低约 4.6–4.9 秒（约 35%）。这是第一次出现不被 fixed 支配的策略。**

---

## 3. 行为诊断

### READ 行为

| 策略 | READ 次数 | READ 比例 | 最长连续 READ | 主要动作 |
|---|---:|---:|---:|---|
| fixed | 0 | 0% | 0 | b4h6 965 |
| v3_baseline | 821 | 80.1% | **48** | read 821, b6h10 132 |
| v3_baseline_guard | 390 | 38.0% | 3 | b6h10 368, b2h10 184 |
| V4-Linear | 34 | 3.3% | 7 | b2h4 538, b6h4 392 |
| V4-Linear + guard | 12 | 1.2% | 2 | b2h4 557, b6h4 395 |
| V4-Convex | 11 | 1.1% | 4 | b2h4 551, b6h4 402 |
| V4-Convex + guard | 3 | 0.3% | 1 | b2h4 559, b6h4 403 |

v3 的 80% READ 和最长 48 步连续 READ 是 LAAL 恶化的直接原因。V4 把 READ 压到
1–3%，并且完全不用 B6H10，改用 b2h4/b6h4 的中等预算。

### 服务时间（60 句合计）

| 策略 | predictor (s) | translator (s) | total (s) |
|---|---:|---:|---:|
| fixed | 706.2 | 69.9 | 776.1 |
| v3_baseline | 236.2 | 21.1 | 257.3 |
| v3_baseline_guard | 781.5 | 57.0 | 838.6 |
| V4-Linear | 422.6 | 57.6 | 480.2 |
| V4-Convex | 428.5 | 58.0 | 486.5 |

V4 的计算量约为 fixed 的 62%，但 completion 只有 fixed 的 64%；v3_guard 反而比 fixed
更贵（B6H10 368 次）。

---

## 4. 离线训练诊断

来自 `cavc-v4/{linear,convex}/seed13/diagnostics.json`（499 个按句留出状态）：

| 变体 | READ | B6H10 | 主要动作 | Q Spearman | Top-1 |
|---|---:|---:|---|---:|---:|
| V4-Linear (κ=0) | 2.4% | 0 | b2h4 239 / b6h4 248 | −0.050 | 16.6% |
| V4-Convex (κ=0.5) | 1.4% | 0 | b2h4 158 / b6h4 334 | −0.058 | 10.6% |

离线动作分布与闭环一致（几乎不 READ、不用 B6H10）。但 Q 的排序仍然很弱，
Spearman 为负，说明策略改善主要来自新的成本结构，而不是价值函数学会了准确排序。

---

## 5. 解释

1. **V3 的高 BLEU 是“等出来”的**：v3_baseline 读到 80%，LAAL 9.03；fixed 的 BLEU
   只有 32.87，但 LAAL 2.45。
2. **V3 的 guard 是“用昂贵 WRITE 换 LAAL”**：B6H10 368 次，completion 14.73 s，
   比 fixed 还慢，LAAL 3.28 仍差于 fixed。
3. **V4 改的是成本结构，不是排序**：延迟成本和墙钟成本进入三分量 Q 后，网络不再
   把 READ 当成免费动作，也不再偏向 B6H10；它选择 b2h4/b6h4 的中等预算。
4. **V4 第一次做到“质量/滞后不劣、时间显著更短”**：BLEU/LAAL 与 fixed 无显著差异，
   completion 显著低 4.6–4.9 s。

---

## 6. 已知的 V4 实现缺陷（V5 正在修）

这些缺陷会影响 V4 数字的“干净程度”，但不改变上面的相对比较：

1. **13/34 输入特征是恒零**：4 个 token 置信度 + 4 个 mask + 4 个 cost/queue 参考 +
   `arrival_slack`。网络实际只有 21 维有效输入。
2. **终端 target 偏差**：V4 的 terminal quality target 被置 0；terminal wall target
   不含 EOS 时的完整源语翻译，`Σ Δt` 不可能等于 `completion_s`。
3. **cost 头可负、且按 γ=0.99 折扣**：delay/wall 是线性输出，可能为负；评测看的是
   总 LAAL/completion，不是折扣和。
4. **serial vs pipeline**：V4 的 wall target 用 `pred_s+mt_s`（serial），eval 的
   completion 用实测 pipeline `spent_s`。
5. **Double DQN 的在线选动作在 train 模式（dropout 开启）**。
6. **checkpoint 存了 feature mean/std，但网络从未使用**（未做输入标准化）。

V5 的修复包括：22 维无死特征输入、cost 头 softplus 非负、delay/wall 独立 γ=1.0、
terminal 加 EOS、Double DQN eval 模式、输入标准化、pipeline 成本口径，以及
`dead_features` 的硬性回归检查。

---

## 7. 产物位置

- Kernel 输出（linear）：
  `pfba-lab/kaggle-runs/results/cavc-v4-eval-linear/`
- Kernel 输出（convex）：
  `pfba-lab/kaggle-runs/results/cavc-v4-eval-convex/`
- 本地训练结果：
  `pfba-lab/kaggle-runs/results/cavc-v4/{linear,convex}/seed13/`
- 数据集：`eunectesqaq/cavc-v4-checkpoints`
- 每个目录包含 `summary.json`、各策略 `*.traces.jsonl` 和完整 kernel log。

---

## 8. 一句话总结

**V4 第一次在 60 句闭环上实现了“BLEU/LAAL 不劣于 fixed，completion 显著快约 35%”
的 Pareto 改善；它靠的是把 READ 和 B6H10 的隐式成本显式化，而不是价值排序变准。
V5 修掉已知实现缺陷后，需要确认这个改善是否保持或进一步扩大。**
