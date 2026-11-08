import json
from types import SimpleNamespace

import pytest

from checkpoint import Journal


SAMPLES = [SimpleNamespace(sentence_id=str(i), source=f"source{i}", reference=f"ref{i}")
           for i in (10, 11)]
METADATA = {"config": {"branches": 4}, "predictor_dtype": "float16"}


def row(sample):
    return {"sentence_id": sample.sentence_id, "source": sample.source,
            "reference": sample.reference, "hypothesis": "x", "replay_s": 0.2}


def test_journal_resumes_mode_prefixes_and_keeps_original_ids(tmp_path):
    journal = Journal(tmp_path, METADATA, SAMPLES, ["taf", "bounded-1"], False)
    journal.append("taf", row(SAMPLES[0]))
    with pytest.raises(ValueError, match="incomplete"):
        journal.require_complete()
    resumed = Journal(tmp_path, METADATA, SAMPLES, ["taf", "bounded-1"], True)
    assert resumed.rows["taf"][0]["sentence_id"] == "10"
    resumed.append("bounded-1", row(SAMPLES[0]))
    resumed.append("taf", row(SAMPLES[1]))
    resumed.append("bounded-1", row(SAMPLES[1]))
    resumed.require_complete()


def test_resume_rejects_precision_source_and_metadata_changes(tmp_path):
    Journal(tmp_path, METADATA, SAMPLES, ["taf"], False)
    with pytest.raises(ValueError, match="metadata"):
        Journal(tmp_path, {**METADATA, "predictor_dtype": "bfloat16"}, SAMPLES, ["taf"], True)
    with pytest.raises(ValueError, match="metadata"):
        Journal(tmp_path, METADATA, SAMPLES[1:], ["taf"], True)


def test_resume_repairs_only_partial_last_record(tmp_path):
    journal = Journal(tmp_path, METADATA, SAMPLES, ["taf"], False)
    journal.append("taf", row(SAMPLES[0]))
    path = tmp_path / "taf.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"sentence_id":')
    resumed = Journal(tmp_path, METADATA, SAMPLES, ["taf"], True)
    resumed.append("taf", row(SAMPLES[1]))
    resumed.require_complete()
    assert len(path.read_text().splitlines()) == 2


def test_resume_refuses_duplicate_or_reordered_original_ids(tmp_path):
    journal = Journal(tmp_path, METADATA, SAMPLES, ["taf"], False)
    with pytest.raises(ValueError, match="order"):
        journal.append("taf", row(SAMPLES[1]))
    journal.append("taf", row(SAMPLES[0]))
    with pytest.raises(ValueError, match="order"):
        journal.append("taf", row(SAMPLES[0]))


def test_existing_results_require_explicit_resume(tmp_path):
    Journal(tmp_path, METADATA, SAMPLES, ["taf"], False)
    with pytest.raises(ValueError, match="resume"):
        Journal(tmp_path, METADATA, SAMPLES, ["taf"], False)

def test_trace_roundtrip_after_checkpoint_preserves_scoring_fields(tmp_path):
    from dataclasses import asdict

    from pipeline import Config, run_sentence
    from run_decoupled import restore_trace

    class Predictor:
        def sample(self, *_):
            return ["x"]

    class Translator:
        def translate_batch(self, sources, prefix):
            return ["target"] * len(sources)

    trace = run_sentence("one two three", "target", Predictor(), Translator(),
                         Config(min_start=2, arrival_gap=0.001), "taf")
    sample = SimpleNamespace(sentence_id="original:42", source="one two three", reference="target")
    journal = Journal(tmp_path, METADATA, [sample], ["taf"], False)
    journal.append("taf", {**row(sample), **asdict(trace)})
    resumed = Journal(tmp_path, METADATA, [sample], ["taf"], True)
    restored = restore_trace(resumed.rows["taf"][0])
    assert restored == trace
