"""FLEURS loading: parallel (source audio, source transcript, English text) triples."""
from parlecho.config import LANGUAGES

FLEURS_CODES = {
    "en": "en_us", "es": "es_419", "fr": "fr_fr",
    "de": "de_de", "hi": "hi_in", "ja": "ja_jp",
    "pt": "pt_br", "it": "it_it",
}


def load_pair(source: str, target: str = "en", n: int = 100, seed: int = 42):
    """Yield dicts: {id, audio (np array), sr, src_transcript, src_raw, tgt_raw}.
    Only sentence ids present in both languages are used."""
    from datasets import load_dataset

    src_ds = load_dataset("google/fleurs", FLEURS_CODES[source], split="test")
    tgt_ds = load_dataset("google/fleurs", FLEURS_CODES[target], split="test")

    # target side: id -> raw (cased, punctuated) text, for BLEU references
    tgt_map = {ex["id"]: ex["raw_transcription"] for ex in tgt_ds}

    src_ds = src_ds.filter(lambda ex: ex["id"] in tgt_map)
    src_ds = src_ds.shuffle(seed=seed).select(range(min(n, len(src_ds))))

    for ex in src_ds:
        yield {
            "id": ex["id"],
            "audio": ex["audio"]["array"],
            "sr": ex["audio"]["sampling_rate"],
            "src_transcript": ex["transcription"],      # normalized, for WER
            "src_raw": ex["raw_transcription"],          # cased, for oracle MT
            "tgt_raw": tgt_map[ex["id"]],                # English reference
        }