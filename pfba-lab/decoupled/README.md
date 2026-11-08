# PFBA prediction and translation pipeline

`run_decoupled.py` measures a paced text replay. A monotonic clock releases each source word at the configured interval. Each sentence has one prediction worker and one translation consumer. Prediction uses the observed source prefix and a seed tied to the sentence and source position. Translation reads the latest committed target prefix and is the only stage that can append to it.

- `taf` processes each available prefix sequentially, samples futures and translates them as one batch.
- `overlap-fifo` schedules the same per-prefix decisions using independent devices and a bounded ready queue. Its hypotheses and content delays should match the sequential baseline when prediction and translation are deterministic on the chosen devices.
- `overlap-latest` allows a producer to skip stale source positions and allows the consumer to discard obsolete ready predictions. Once the full source arrives, the next translation uses the true source. An already-running prediction can finish after the final translation. The trace records final output and worker drain separately.
- `bounded-1`, `bounded-2`, and `bounded-3` cap how many observed source positions can be skipped between successive translations. They wait for the configured number of completed initial translation steps and committed target words before skipping. Once a prediction enters the ready queue it is consumed in order, including at sentence end. The scheduler processes the penultimate source prefix before the true-source flush when that prefix is eligible for speculation.
- A mode ending in `-anchor` adds the real observed source prefix to the translation batch. `vote` gives its translated suffix one extra vote. `veto` only commits the words shared by the speculative consensus and the anchor. The anchor sees the same observed prefix as its prediction. The terminal full-source translation is unchanged.

The default worker holds at most one ready result plus one prediction in flight. Translation candidates are decoded on the same device and with the same settings across modes. Benchmark order rotates by sentence to reduce ordering effects. Wall time excludes model loading, warm-up and metric scoring. It includes actual pacing, host scheduling and synchronized GPU work. This is controlled text replay. The prediction and translation span overlap measures concurrent host-stage intervals, which include transfers and synchronization.

`--prediction-batch-size` passes a common prediction microbatch size to the shared predictor adapter. Its default is zero. The shared adapter must implement this attribute for it to limit memory. Every completed sentence and mode is written durably to its JSONL trace. Include `checkpoint.py` in the Kaggle package. Resume an interrupted experiment with `--resume` and the same output directory and arguments. The checkpoint checks configuration, precision, model revisions, GPU/runtime identity, code hashes and original sentence IDs and contents before accepting completed results. A partially appended final record is rolled back and rerun. Scoring begins after every requested sentence and mode has a valid record.

`--cpu-predictor-embedding` keeps an untied input embedding on CPU in its stored dtype and transfers lookup results to the input device. Package `cpu_embedding.py` when enabling it. For Qwen2.5-7B, the 152064 by 3584 FP16 input table uses about 1.015 GiB. A zero-element execution parameter preserves the model device and dtype reported to generation. Apply this after loading the model and keep the model placement fixed afterward. The flag leaves batch size, random seed sequence, transformer layers and output projection unchanged. Validate real-checkpoint candidate parity before using the path in a matched experiment. The runtime wrapper is intended for inference and is not an original-checkpoint serialization format.

## Local validation

```powershell
Set-Location pfba-lab/decoupled
& 'D:/MINICONDA/envs/py312/python.exe' -m pytest -q
& 'D:/MINICONDA/envs/py312/python.exe' run_decoupled.py --output-dir outputs/dev --dry-run
```

## GPU benchmark

Place `pipeline.py`, `run_decoupled.py` and the current `paired/run_pfba.py` next to each other in a Kaggle package. Use the same model and metric dependencies as the paired experiment. For two available GPUs run:

```bash
python -u run_decoupled.py --predictor large --translator large \
  --predictor-device cuda:0 --translator-device cuda:1 --split dev --limit 12 \
  --modes taf overlap-fifo overlap-latest overlap-fifo-anchor \
  --anchor-policy vote --anchor-threshold 0.6 --output-dir /kaggle/working/decoupled
```

Use `--modes taf bounded-1 bounded-2 bounded-3` to compare the skip caps in the same run. Set `--bounded-initial-steps` and `--bounded-initial-words` to tune the initial guard. `--bounded-threshold` applies only to bounded modes. A separately provided `--anchor-threshold` takes precedence in the anchor modes. Use development runs to select these values and verify paired content delay before locking configurations. The JSONL traces preserve sentence identity, hypotheses, committed source positions, candidate hashes and actual stage intervals. The summary includes BLEU, COMET, LAAL, first output, final output, worker drain, overlap and work counts.

The option --bounded-skip-every N throttles actual skips after the initial guards. The first skip is eligible as soon as those guards hold. After a skip, another skip requires N additional completed non-terminal translation calls. A prediction that advances to the next consecutive source position consumes no skip allowance. The default value 1 preserves the earlier scheduling behavior. The named modes bounded-1-every2 and bounded-1-every3 select skip cap 1 and a cadence of 2 or 3 regardless of the CLI cadence, so one run can compare --modes taf bounded-1 bounded-1-every2 bounded-1-every3. Prediction spans record the completed translation count and skipped positions at submission time. Source availability, the queue limit and the per-translation skip cap still apply.
