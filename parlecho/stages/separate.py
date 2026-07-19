"""Stage 1: source separation. Splits audio into vocals and accompaniment
using Demucs, so translation only touches dialogue and music survives intact."""
import gc
from pathlib import Path

import torch
import torchaudio


def separate(
    input_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    device: str = "cpu",
) -> dict[str, Path]:
    """Run Demucs 2-stem separation. Returns paths to vocals + accompaniment WAVs."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import AudioFile

    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Load audio at the model's expected sample rate and channel count
    wav = AudioFile(input_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()  # normalize like the demucs CLI does

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],          # add batch dim
            device=device,
            shifts=1,
            split=True,         # chunked processing, keeps memory bounded
            overlap=0.25,
            progress=True,
        )[0]                    # drop batch dim

    sources = sources * ref.std() + ref.mean()  # denormalize

    # htdemucs outputs 4 stems: drums, bass, other, vocals
    stem_names = model.sources
    vocals_idx = stem_names.index("vocals")

    vocals = sources[vocals_idx]
    accompaniment = torch.stack(
        [sources[i] for i in range(len(stem_names)) if i != vocals_idx]
    ).sum(0)

    paths = {}
    for name, tensor in [("vocals", vocals), ("accompaniment", accompaniment)]:
        out = output_dir / f"{name}.wav"
        torchaudio.save(str(out), tensor.cpu(), model.samplerate)
        paths[name] = out

    # Free memory before the next stage loads its model
    del model, sources, wav
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return paths