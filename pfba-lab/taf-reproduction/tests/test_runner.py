"""End-to-end mock pipeline, metrics, and artifact/manifest audit tests."""
import json
from pathlib import Path

import pytest

from taf.bleu import corpus_bleu
from taf.config import StreamingConfig
from taf.mocks import MockLLM, MockMT, make_vocab
from taf.runner import (
    build_manifest,
    run_dataset,
    word_level_delays,
    write_run_artifacts,
)

MAPPING = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
VOCAB = make_vocab(["A", "B", "C", "D", "E"])


def make_pipeline():
    # At "a" the LLM predicts the rest of the sentence 8/10 times, so TAF
    # commits the full 5-token hypothesis at read position 1.  Subsequent
    # read steps predict nothing and the committed prefix is preserved.
    llm = MockLLM(table={
        "a": ["b c d e"] * 8 + ["b c d"] * 2,
        "a b c d e": [""] * 10,
    })
    mt = MockMT(mapping=MAPPING, vocab=VOCAB)
    cfg = StreamingConfig(policy="taf", n=10, tau=0.6, cuda_sync=False,
                          max_new_tokens=64)
    return llm, mt, cfg


def test_run_dataset_end_to_end_mock():
    llm, mt, cfg = make_pipeline()
    run = run_dataset(["a b c d e"], ["A B C D E"], llm, mt, cfg)
    sent = run.sentences[0]

    assert sent.hypothesis == "A B C D E"
    assert sent.metrics["n_source_units"] == 5
    assert sent.metrics["n_target_words"] == 5
    # All five tokens committed at read_pos=1; S=T=5 -> slope 1.
    # SimulEval does not clamp: terms 1,0,-1,-2,-3 -> mean -1.0.
    assert sent.metrics["laal"] == pytest.approx(-1.0)
    assert sent.metrics["laal_cutoff"] == 5
    assert sent.metrics["laal_target_length"] == 5
    assert run.summary["bleu"]["score"] == pytest.approx(100.0)
    assert run.summary["comet"]["score"] is None
    assert run.summary["comet"]["status"] == "disabled"
    assert run.summary["timing"]["n_steps"] == 5
    assert run.summary["policy"]["forced_committed_prefix"] is True
    json.dumps(run.to_dict())


def test_laal_cutoff_stops_at_first_full_source_emission():
    # The full-source anticipation step emits A; the later fresh EOS decode
    # emits B C at the same (full-source) delay.  SimulEval's cutoff is the
    # first full-source emission, so B C must not enter LAAL.
    llm = MockLLM(table={"a": ["c"] * 4, "a b": ["d"] * 4})

    vocab = make_vocab(["A", "B", "C"])

    def variants(source, committed, count):
        if source == "a c":
            return [[]]
        if not committed:
            return [[vocab["A"]]]
        return [[vocab["B"], vocab["C"]]]

    mt = MockMT(vocab=vocab, variant_fn=variants)
    cfg = StreamingConfig(
        policy="ralcp_taf", n=4, beam_width=4, ralcp_fraction=0.5,
        min_start=1, cuda_sync=False, max_new_tokens=64,
    )
    run = run_dataset(["a b"], ["A B C"], llm, mt, cfg)
    sent = run.sentences[0]

    assert sent.hypothesis == "A B C"
    assert sent.word_delays == [2, 2, 2]
    # Cutoff is one target word (A), not all three emissions.
    assert sent.metrics["laal_cutoff"] == 1
    assert sent.metrics["laal_reference_length"] == 3
    # T = max(3, 1) = 3; slope (2-1)/(3-1); only delay 2 is kept.
    assert sent.metrics["laal"] == pytest.approx(2.0 / 3.0)
    assert sent.metrics["laal_no_cutoff"] == pytest.approx(1.5)


def test_word_level_delays_with_subword_markers():
    tokens = ["▁The", "▁cat", "s", "▁sat"]
    delays = [1, 2, 2, 3]
    assert word_level_delays(tokens, delays, word_level=False) == [1, 2, 3]
    assert word_level_delays(["A", "B"], [1, 2], word_level=True) == [1, 2]


def test_artifacts_and_manifest(tmp_path):
    llm, mt, cfg = make_pipeline()
    run = run_dataset(["a b c d e"], ["A B C D E"], llm, mt, cfg)
    manifest = build_manifest(
        cfg,
        llm_name="meta-llama/Llama-2-7b-hf",
        mt_name="Helsinki-NLP/opus-mt-de-en",
        comet_name=None,
        data_path="data/sample_de_en.tsv",
        argv=["reproduce_de_en.py"],
    )
    out = write_run_artifacts(run, manifest, tmp_path / "run")

    for name in ("sources.txt", "references.txt", "hypotheses.txt",
                 "traces.jsonl", "per_sentence.jsonl", "summary.json", "manifest.json"):
        assert (out / name).exists(), name

    assert (out / "hypotheses.txt").read_text(encoding="utf-8").strip() == "A B C D E"
    trace_lines = (out / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # five read steps + one flush event
    assert len(trace_lines) == 6
    assert json.loads(trace_lines[-1])["event"] == "flush"

    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved["classification"]["llm"].startswith("paper-exact")
    assert saved["classification"]["mt"] == "substitution"
    assert saved["classification"]["comet"] == "none"
    assert "local checkout" in saved["paper"]["official_code"]
    assert saved["paper"]["author_checkout"]["commit"].startswith("99e373a")
    assert saved["notes"]


def test_manifest_xcomet_is_paper_cited():
    cfg = StreamingConfig()
    manifest = build_manifest(
        cfg,
        llm_name="meta-llama/Llama-2-7b-hf",
        mt_name="transformer-big-de-en",
        comet_name="Unbabel/XCOMET-XXL",
        data_path="wmt15/newstest2015",
    )
    assert manifest["classification"]["mt"].startswith("paper-exact")
    assert manifest["classification"]["comet"].startswith("paper-cited")


def test_identical_three_word_corpus_is_zero_4gram_bleu():
    # Default corpus BLEU goes up to 4-grams.  A 3-word hypothesis/reference
    # has no 4-gram matches, so 100 is impossible; fixtures asserting a
    # perfect score must use at least four identical words.
    assert corpus_bleu(["a b c"], ["a b c"])["score"] == pytest.approx(0.0)
    assert corpus_bleu(["a b c d"], ["a b c d"])["score"] == pytest.approx(100.0)


def test_author_dev_data_pair_is_readable_and_smoke_runs():
    upstream = Path(__file__).resolve().parents[2] / "taf-upstream" / "data" / "de_en"
    src_path, ref_path = upstream / "dev.de", upstream / "dev.en"
    if not src_path.exists() or not ref_path.exists():
        pytest.skip(f"author checkout not available at {upstream}")
    sources = src_path.read_text(encoding="utf-8").splitlines()[:3]
    references = ref_path.read_text(encoding="utf-8").splitlines()[:3]
    assert len(sources) == len(references) == 3
    assert all(sources) and all(references)

    llm = MockLLM(default=lambda key, n: [""] * n)
    words = {word for source in sources for word in source.split()}
    mt = MockMT(mapping={}, vocab=make_vocab(words))
    cfg = StreamingConfig(policy="ralcp_taf", n=4, beam_width=4,
                          cuda_sync=False, max_new_tokens=64)
    run = run_dataset(sources, references, llm, mt, cfg)

    assert len(run.sentences) == 3
    assert all(s.hypothesis for s in run.sentences)
    assert run.summary["laal"] is not None
    assert run.summary["bleu"]["score"] is not None


def test_cli_pairs_author_dev_files(tmp_path):
    upstream = Path(__file__).resolve().parents[2] / "taf-upstream" / "data" / "de_en"
    src_path, ref_path = upstream / "dev.de", upstream / "dev.en"
    if not src_path.exists() or not ref_path.exists():
        pytest.skip(f"author checkout not available at {upstream}")

    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_de_en.py"
    spec = importlib.util.spec_from_file_location("reproduce_de_en", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "run"
    rc = module.main([
        "--data", str(src_path),
        "--ref-data", str(ref_path),
        "--llm-backend", "mock",
        "--mt-backend", "mock",
        "--policy", "ralcp_taf",
        "--n", "4",
        "--beam-width", "4",
        "--max-sentences", "2",
        "--output-dir", str(out),
        "--quiet",
    ])

    assert rc == 0
    assert (out / "summary.json").exists()
    assert len((out / "hypotheses.txt").read_text(encoding="utf-8").splitlines()) == 2
