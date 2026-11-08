import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6",
     "sacrebleu==2.5.1", "sentencepiece", "accelerate", "sacremoses",
     "entmax", "jsonargparse==3.13.1", "torchmetrics==0.10.3", "pytorch-lightning"],
    check=True,
)
# Keep the Python 3.13 Kaggle image's NumPy ABI. COMET 2.2.7 declares a NumPy
# 1.x cap, for which upstream NumPy publishes no CPython 3.13 wheel.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "unbabel-comet==2.2.7"],
    check=True,
)
worker = r"""
import importlib.metadata
import json
from pathlib import Path

import numpy
import torch

assert torch.cuda.is_available(), "Kaggle GPU is required"
print("NumPy:", numpy.__version__, "torch:", torch.__version__, flush=True)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

source = "Heute ist das Wetter"
qwen = "Qwen/Qwen2.5-0.5B"
marian = "Helsinki-NLP/opus-mt-de-en"
with torch.inference_mode():
    tokenizer = AutoTokenizer.from_pretrained(qwen)
    model = AutoModelForCausalLM.from_pretrained(qwen, torch_dtype=torch.float16).to("cuda").eval()
    inputs = tokenizer(source, return_tensors="pt").to("cuda")
    future = tokenizer.decode(model.generate(**inputs, max_new_tokens=3)[0], skip_special_tokens=True)
    del model
    tokenizer = AutoTokenizer.from_pretrained(marian)
    model = AutoModelForSeq2SeqLM.from_pretrained(marian, torch_dtype=torch.float16).to("cuda").eval()
    inputs = tokenizer(future, return_tensors="pt").to("cuda")
    hypothesis = tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True)
    del model
print("Translation:", hypothesis, flush=True)
from comet import download_model, load_from_checkpoint
metric = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
score = metric.predict(
    [{"src": future, "mt": hypothesis, "ref": "The weather is nice today."}], batch_size=1, gpus=1,
)
payload = {
    "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "future": future,
    "translation": hypothesis,
    "preflight_comet": float(score.system_score),
    "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "sacrebleu", "unbabel-comet")},
}
Path("/kaggle/working/preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(payload, flush=True)

"""
subprocess.run([sys.executable, "-u", "-c", worker], check=True)
