"""Stage 4: translation. NLLB-200 translates each segment's text
into the target language. Runs on MPS."""
import gc
from pathlib import Path

from parlecho.config import lang
from parlecho.stages.asr import Segment


def translate(
    segments: list[Segment],
    source_lang: str,          # our registry code, e.g. "es"
    target_lang: str,          # e.g. "en"
    model_name: str = "facebook/nllb-200-distilled-600M",
    device: str = "cpu",
) -> list[Segment]:
    """Replace each segment's text with its translation. Returns new segments."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src = lang(source_lang, "nllb")
    tgt = lang(target_lang, "nllb")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    translated: list[Segment] = []
    batch_size = 8
    texts = [s.text for s in segments]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
                max_new_tokens=200,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for seg, text in zip(segments[i : i + batch_size], decoded):
                translated.append(
                    Segment(start=seg.start, end=seg.end,
                            text=text.strip(), speaker=seg.speaker)
                )

    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return translated