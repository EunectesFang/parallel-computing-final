# 并行计算期末：德语到英语的同声传译

这个仓库是课程期末项目。任务是在源句还没说完时把德语译成英语，基线是 TAF：每一步采样若干条德语未来，翻译后用 RALCP 投票，交出大家一致的英语前缀。实验代码、Kaggle 任务、结果和文字记录都在这里。上游 TAF 以子模块放在 `pfba-lab/taf-upstream`。

## 顶层

| 路径 | 里面有什么 |
|---|---|
| `pfba-lab/` | 实验代码、打包好的 Kaggle 脚本、结果 JSON 和日志 |
| `现有改进和未来思路/` | 已经跑完的改进记录、后面的方案，以及还没查清的问题 |
| `PPT/` | 选题报告的逐页大纲，以及一份可打开的 `PPT.html` |
| `论文/` | TAF 论文 PDF（`2410.22499v2.pdf`）、文献阅读笔记和一份研究方案 PDF |
| `.gitmodules` | 声明 `pfba-lab/taf-upstream` 指向 `https://github.com/owaski/TAF.git` |

`.pytest_cache/` 和各目录下的 `__pycache__/` 是本地测试缓存，不进版本库。

## `现有改进和未来思路/`

| 文件或文件夹 | 里面有什么 |
|---|---|
| `现有改进.md` | 已经写进代码并且有数字的改进：请求复用、前缀 KV、双卡重叠、跳步、动态调控，以及 2026-11-09 的三场一折 |
| `未来思路.md` | 早停、跨步 KV、双卡调度和 101 个动作的调控器要怎么合成一套。第 4.8 节是本周实际跑法和已经出来的数 |
| `bug/` | 7B 上加了译文置信度、译文却和没加时逐位相同。`问题.md` 写已排除的接错和还没排除的点；两场 Kaggle 日志和 `summary.json` 放在同目录 |

## `pfba-lab/`

| 路径 | 里面有什么 |
|---|---|
| `taf-upstream/` | 上游 TAF 子模块。`agents/base.py` 里 B、H 的默认上限是 10。`data/de_en/` 是开发集 3003 句和测试集 2169 句 |
| `taf-reproduction/` | 对 TAF 的本地复现：RALCP、LAAL、BLEU、策略和 Hugging Face 后端。`src/taf/` 是库，`tests/` 是测试，`scripts/reproduce_de_en.py` 是入口 |
| `paired/` | 同输出对照用的运行器 `run_pfba.py`：采样、Marian / Tower 翻译、投票早停、LAAL。`pfba-lab/paired_evidence.py` 做句子配对的重抽样区间 |
| `decoupled/` | 预测和翻译分到两张卡。`pipeline.py` 是先入先出等调度，`checkpoint.py` 断点续跑，`cpu_embedding.py` 把 7B 输入词表放到 CPU |
| `kv-experiment/` | 旧的前缀 KV（`kv_prefix.py`）。只复用提示词的 KV，生成预算没减，FP16 上候选还会变 |
| `joint-gate/` | 早期动态调控。动作只有四个 (B, H) 再加继续听。`joint_gate.py` 打标签和训练，`sweep_gate.py` 扫权重 |
| `week/` | 本周一折调控器。`week_run.py` 标注、训练、留出集闭环。听的收益是少交出去的正确词数；动作头先判听还是写，再选 B 和 H；决策前可读翻译模型对下一词的置信度。`package_kernel.py` 把脚本打成 Kaggle 入口，`test_week_run.py` 是单元测试 |
| `kaggle-preflight/` | 推上 Kaggle 之前的环境和数据检查 |
| `kaggle-runs/` | 每次推上去的私有脚本。每个子目录是一场，里面是 `kaggle_entry.py` 和 `kernel-metadata.json`。目录名和 Kaggle 上的 kernel 名一致 |
| `results/` | 已经下回来的日志和 `summary.json`。每个子目录对应一场。`week-fold1/`、`week-fold2/`、`week-confidence/` 是 2026-11-09 的三场一折 |
| `docs/` | 实验笔记和计划草稿 |
| `prepare_kernel.py`、`prepare_decoupled_kernel.py`、`prepare_gate_kernel.py` | 把对应实验打包成 `kaggle-runs/` 里的一场 |
| `INTEGRATION.md` | 这些目录怎么接在一起 |

`kaggle-runs/` 里和本周直接相关的三场：

| 目录 | Kaggle | 做什么 |
|---|---|---|
| `pfba-week-fold1` | `eunectesqaq/pfba-week-fold1` | 101 个动作、温度 1 的 softmax。策略几乎一直听到句末 |
| `pfba-week-fold2` | `eunectesqaq/pfba-week-fold2` | 听的收益改成 −m，动作头拆成两段，从空前缀采状态。不再等到句末，BLEU 下降 |
| `pfba-week-confidence` | `eunectesqaq/pfba-week-confidence` | 在 fold2 上每步多读译文置信度。小模型 BLEU 略回升；7B 前 64 句的译文和 fold2 逐位相同 |
