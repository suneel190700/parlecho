"""CTranslate2 int8 backend for the translate stage. Same Segment-in,
Segment-out interface as translate.py, so the pipeline can swap backends."""
import gc
from pathlib import Path

from parlecho.config import lang
from parlecho.stages.asr import Segment

DEFAULT_MODEL_DIR = Path("models/nllb-600M-ct2")
TOKENIZER_NAME = "facebook/nllb-200-distilled-600M"


def translate_ct2(
    segments: list[Segment],
    source_lang: str,
    target_lang: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str = "cpu",              # "cuda" on GPU; MPS not supported by CT2
    compute_type: str = "int8",
    translator=None,                  # pass a loaded Translator to reuse
    beam_size: int = 4,
) -> list[Segment]:
    import ctranslate2
    from transformers import AutoTokenizer

    src = lang(source_lang, "nllb")
    tgt = lang(target_lang, "nllb")

    owns = translator is None
    if owns:
        translator = ctranslate2.Translator(
            str(model_dir), device=device, compute_type=compute_type
        )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, src_lang=src)

    texts = [s.text for s in segments]
    tokenized = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in texts
    ]

    results = translator.translate_batch(
        tokenized,
        target_prefix=[[tgt]] * len(tokenized),
        max_batch_size=16,
        beam_size=beam_size,
    )

    out: list[Segment] = []
    for seg, res in zip(segments, results):
        # hypothesis starts with the target-language token — drop it
        tokens = res.hypotheses[0][1:]
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
        )
        out.append(Segment(start=seg.start, end=seg.end,
                           text=text.strip(), speaker=seg.speaker))

    if owns:
        del translator
        gc.collect()

    return out