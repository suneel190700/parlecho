"""Parlecho configuration: devices, paths, language registry."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_CACHE = PROJECT_ROOT / "models"


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DeviceConfig:
    """Per-stage devices. CTranslate2 (faster-whisper) has no MPS backend,
    and XTTS-v2 is unstable on MPS, so both fall back to CPU on Mac."""
    separate: str = field(default_factory=best_device)   # Demucs
    asr: str = "cuda" if torch.cuda.is_available() else "cpu"
    diarize: str = field(default_factory=best_device)    # pyannote
    translate: str = field(default_factory=best_device)  # NLLB
    tts: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PipelineConfig:
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    whisper_model: str = "small"          # dev default; "large-v3" on rented GPU
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    demucs_model: str = "htdemucs"
    sample_rate: int = 44100
    tts_sample_rate: int = 24000          # XTTS native output rate
    max_stretch: float = 1.35             # clamp for time-stretch fitting
    min_stretch: float = 0.75


# Language registry: one row per language, mapping codes across backends.
# whisper: ISO 639-1 | nllb: FLORES-200 code | xtts: XTTS-v2 code (None = unsupported)
LANGUAGES = {
    "en": {"whisper": "en", "nllb": "eng_Latn", "xtts": "en"},
    "es": {"whisper": "es", "nllb": "spa_Latn", "xtts": "es"},
    "fr": {"whisper": "fr", "nllb": "fra_Latn", "xtts": "fr"},
    "de": {"whisper": "de", "nllb": "deu_Latn", "xtts": "de"},
    "hi": {"whisper": "hi", "nllb": "hin_Deva", "xtts": "hi"},
    "ja": {"whisper": "ja", "nllb": "jpn_Jpan", "xtts": "ja"},
    "pt": {"whisper": "pt", "nllb": "por_Latn", "xtts": "pt"},
    "it": {"whisper": "it", "nllb": "ita_Latn", "xtts": "it"},
}


def lang(code: str, backend: str) -> str:
    entry = LANGUAGES.get(code)
    if entry is None:
        raise ValueError(f"Language '{code}' not in registry")
    mapped = entry[backend]
    if mapped is None:
        raise ValueError(f"Language '{code}' unsupported by {backend}")
    return mapped