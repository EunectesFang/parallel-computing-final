"""Durable per-mode progress with strict experiment identity checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Journal:
    def __init__(self, directory, metadata, samples, modes, resume):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.samples = samples
        self.rows = {mode: [] for mode in modes}
        sample_identity = [
            {"sentence_id": sample.sentence_id, "source": sample.source, "reference": sample.reference}
            for sample in samples
        ]
        if len({sample.sentence_id for sample in samples}) != len(samples):
            raise ValueError("sample IDs must be unique")
        identity = {
            "version": 1, "metadata": metadata, "modes": list(modes),
            "samples_sha256": hashlib.sha256(json.dumps(sample_identity).encode()).hexdigest(),
            "sample_ids": [sample.sentence_id for sample in samples],
        }
        manifest = self.directory / "checkpoint.json"
        if manifest.exists():
            if not resume:
                raise ValueError("existing experiment requires --resume")
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("resume metadata or selected samples differ")
        else:
            if resume:
                raise ValueError("resume checkpoint.json is missing")
            if ((self.directory / "summary.json").exists()
                    or any((self.directory / f"{mode}.jsonl").exists() for mode in modes)):
                raise ValueError("existing traces require a valid resume checkpoint")
            atomic_json(manifest, identity)

        for mode in modes:
            path = self.directory / f"{mode}.jsonl"
            if not path.exists():
                continue
            data = path.read_bytes()
            valid_end = data.rfind(b"\n") + 1
            # An append interrupted before its final newline is not committed.
            for line in data[:valid_end].splitlines():
                record = json.loads(line)
                self._validate_next(mode, record)
                self.rows[mode].append(record)
            if valid_end != len(data):
                with path.open("r+b") as handle:
                    handle.truncate(valid_end)
                    handle.flush()
                    os.fsync(handle.fileno())

    def _validate_next(self, mode, record):
        index = len(self.rows[mode])
        if index >= len(self.samples):
            raise ValueError("trace order contains too many samples")
        sample = self.samples[index]
        if (record["sentence_id"], record["source"], record["reference"]) != (
            sample.sentence_id, sample.source, sample.reference
        ):
            raise ValueError("trace order or original sample content differs")

    def append(self, mode, record):
        self._validate_next(mode, record)
        data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with (self.directory / f"{mode}.jsonl").open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[mode].append(record)

    def require_complete(self):
        if any(len(rows) != len(self.samples) for rows in self.rows.values()):
            raise ValueError("incomplete experiment cannot be scored")
