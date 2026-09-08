"""Route ML/download caches to the project .cache folder."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CACHE = PROJECT / ".cache"

_DIRS = (
    CACHE / "pip",
    CACHE / "tmp",
    CACHE / "torch",
    CACHE / "huggingface",
    CACHE / "datasets",
    CACHE / "easyocr",
    CACHE / "kaggle",
)

for d in _DIRS:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))
os.environ.setdefault("TMP", str(CACHE / "tmp"))
os.environ.setdefault("TEMP", str(CACHE / "tmp"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "huggingface" / "transformers"))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(CACHE / "easyocr"))
os.environ.setdefault("MODULE_PATH", str(CACHE / "easyocr"))
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(CACHE / "kaggle"))

HF_DATASETS_CACHE = CACHE / "datasets"
HF_HUB_CACHE = CACHE / "huggingface" / "hub"
EASYOCR_CACHE = CACHE / "easyocr"
TORCH_HOME = CACHE / "torch"
