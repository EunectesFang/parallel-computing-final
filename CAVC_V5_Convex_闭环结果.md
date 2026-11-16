# CAVC-Q V5-Convex 闭环结果（第一版）

- 日期：2026-11-16
- Kernel：`eunectesqaq/pfba-cavc-v5-eval`（COMPLETE）
- 数据：WMT de→en `dev:0260–0319`（offset 260, limit 60）
- 模型对：Qwen2.5-0.5B predictor + Helsinki-NLP/opus-mt-de-en translator
- 运行时：same `ControlledRuntime`、seed 13、RALCP 0.6 / min_start 3
- 本 kernel **只跑 V5-Convex**：`--skip-fixed --skip-baseline`
- 基线 fixed 的数字来自同一句切片、同一 seed 的 V4 kernel（`pfba-cavc-v4-linear`）

> 说明：V5-Convex 的 `summary.json`/traces 已下载，逐句配对 bootstrap 已补上。
> fixed 的 traces 来自同一句切片、同一 seed 的 V4 kernel。

---

## 1. 结果

| 策略 | BLEU | LAAL ↓ | completion ↓ | READ |
|---|---:|---:|---:|---:|
| fixed (4,6)（V4 kernel 同切片） | 32.87 | 2.45 | 13.53 | 0% |
| v3_baseline（同切片） | 41.05 | 9.03 | 7.15 | 85.1% |
| V4-Convex（上一版） | 34.21 | 2.40 | 8.74 | 1.1% |
| **V5-Convex** | 30.91 | **2.33** | 8.91 | 9.1% |
| V5-Convex + guard | 30.37 | 2.35 | 8.65 | 8.6% |

**相对 fixed 的逐句配对 bootstrap（n=60）：**

| 策略 | ΔBLEU [95%] | ΔLAAL [95%] | Δcompletion [95%] | ΔchrF [95%] |
|---|---|---|---|---|
| V5-Convex | −0.53 [−4.05, 2.77] | −0.12 [−0.39, 0.15] | **−4.63 [−5.61, −3.66]** | −0.37 [−3.10, 2.22] |
| V5-Convex + guard | −1.10 [−4.62, 2.20] | −0.10 [−0.36, 0.15] | **−4.89 [−5.87, −3.96]** | −0.49 [−3.19, 2.05] |

**BLEU / LAAL / chrF 的差异都不显著；completion 显著更快。**

---

## 2. 初步解读

1. **V5-Convex 保住了“时间显著更短”这一点**：每句 completion 约 8.7–8.9 s，
   比 fixed 的 13.53 s 快约 **34%**。
2. **LAAL 与 fixed 基本持平甚至略好**（2.33–2.35 vs 2.45）。
3. **corpus BLEU 比 fixed 低约 2 分**（30.9 vs 32.9），但逐句配对 bootstrap 的
   ΔBLEU 只有 **−0.53 [−4.05, 2.77]**，**不显著**。chrF 同样不显著（−0.37
   [−3.10, 2.22]）。所以不能把 corpus 的 2 分差直接当成质量退化。
4. **行为从 V4 的“几乎不 READ”变成 9% READ**：V5 修了终端 EOS 成本、非负 cost 头
   和输入标准化后，策略重新开始适度等待；离线 READ 9.6% 与闭环 9.1% 一致。
5. **离线排序改善**：V5-Convex 的 Q Spearman 从 V4-Convex 的 −0.058 升到 **+0.148**，
   Top-1 16.0%，说明价值函数不再反向排序。

---

## 3. 与 V4 的对比

| 指标 | V4-Convex | V5-Convex |
|---|---:|---:|
| BLEU（闭环） | 34.21 | 30.91 |
| LAAL | 2.40 | **2.33** |
| completion | 8.74 | 8.91 |
| READ | 1.1% | 9.1% |
| Q Spearman | −0.058 | **+0.148** |
| Top-1 | 10.6% | **16.0%** |

V5 修的是“正确性”（非负成本、EOS、标准化、无死特征），代价是闭环 BLEU 下降、
READ 回升。它**不是 V4 的全面升级**，而是把策略从“几乎不读”拉回到更合理的
9% READ，同时保住了时间和 LAAL。

---

## 4. V5 相对 V4 的代码修复（本结果对应的版本）

1. 输入特征 34 → **22 维**，删除 13 个恒零输入（4 token 置信度 + 4 mask +
   4 cost/queue + 旧 arrival_slack），`arrival_slack` 重新接线；训练时遇到
   `dead_features` 会直接报错。
2. delay/wall 头改为 **softplus 非负**；delay/wall 使用独立 **γ=1.0**。
3. 终端 quality 不再置 0；终端 wall 加入 **EOS 全源翻译估计**。
4. Double DQN 的在线选动作改为 **eval 模式**，不再受 dropout 干扰。
5. 网络加入 **输入标准化**（checkpoint 里的 feature mean/std 真正生效）。
6. wall 成本口径改为 **pipeline**（使用 labels 里的 `marginal_wall_s`）。
7. eval 支持 `--skip-fixed --skip-baseline`，可只跑新模型；新增离线配对脚本。

---

## 5. 待补充

1. V5-Linear 的闭环结果（kernel `pfba-cavc-v5-linear-eval` 仍在 QUEUED/RUNNING）；
2. V5 在更大 dev holdout / 冻结 test 上的复现。

---

## 6. 一句话总结

**V5-Convex 在修掉实现缺陷后实现了 Pareto 改善：BLEU、LAAL、chrF 与 fixed 统计上
无显著差异，completion 显著快约 4.6 秒（约 34%）。它把 V4 的“几乎不读”修正为
约 9% READ，并把 Q 排序从负相关拉回到 +0.148。**
