"""
Augmentation pipeline for triplets.

Critical invariant: every transform applied to mixture is applied identically
to target using the same RNG state, so mixture == target + rest at all times.
"""
import random
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).cpu().numpy().astype(np.float32)


def _to_tensor(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(a).unsqueeze(0).to(device)


def augment_triplet(
    mixture: torch.Tensor,
    target: torch.Tensor,
    stem_name: str,
    sr: int,
    config: dict,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply identical augmentations to mixture and target.

    Args:
        mixture:   (1, T) float32
        target:    (1, T) float32
        stem_name: instrument name for pitch-skip logic
        sr:        sample rate
        config:    augmentation sub-dict
        rng:       optional seeded Random instance; created if None
    Returns:
        aug_mixture, aug_target: (1, T) each
    """
    if rng is None:
        rng = random.Random()

    device = mixture.device
    mix_np = _to_numpy(mixture)
    tgt_np = _to_numpy(target)

    # --- pitch shift ---
    if rng.random() < config["pitch_shift_prob"]:
        skip = stem_name.lower() in [s.lower() for s in config["pitch_skip_stems"]]
        if not skip:
            semitones = rng.uniform(
                -config["pitch_shift_range_semitones"],
                config["pitch_shift_range_semitones"],
            )
            import pyrubberband as rb
            mix_np = rb.pitch_shift(mix_np, sr, semitones, rbargs={"-F": ""})
            tgt_np = rb.pitch_shift(tgt_np, sr, semitones, rbargs={"-F": ""})

    # --- time stretch ---
    if rng.random() < config["time_stretch_prob"]:
        lo, hi = config["time_stretch_range"]
        factor = rng.uniform(lo, hi)
        import pyrubberband as rb
        mix_np = rb.time_stretch(mix_np, sr, factor)
        tgt_np = rb.time_stretch(tgt_np, sr, factor)
        # Trim or pad back to original length
        orig_len = mixture.shape[-1]
        if mix_np.shape[-1] >= orig_len:
            mix_np = mix_np[:orig_len]
            tgt_np = tgt_np[:orig_len]
        else:
            pad = orig_len - mix_np.shape[-1]
            mix_np = np.pad(mix_np, (0, pad))
            tgt_np = np.pad(tgt_np, (0, pad))

    # --- volume jitter ---
    if rng.random() < config["volume_jitter_prob"]:
        db = rng.uniform(-config["volume_jitter_db"], config["volume_jitter_db"])
        gain = 10 ** (db / 20.0)
        mix_np = mix_np * gain
        tgt_np = tgt_np * gain

    # --- effects chain ---
    if rng.random() < config["effects_chain_prob"]:
        from pedalboard import Pedalboard, Reverb, Delay, Chorus, Phaser
        effects = []
        if rng.random() < 0.40:
            effects.append(Reverb(room_size=rng.uniform(0.1, 0.6)))
        if rng.random() < 0.25:
            effects.append(Delay(delay_seconds=rng.uniform(0.05, 0.3),
                                 feedback=rng.uniform(0.1, 0.5)))
        if rng.random() < 0.20:
            effects.append(Chorus(rate_hz=rng.uniform(0.5, 3.0),
                                  depth=rng.uniform(0.1, 0.5)))
        if rng.random() < 0.15:
            effects.append(Phaser(rate_hz=rng.uniform(0.5, 2.0)))
        if rng.random() < 0.10:
            # flanger = Chorus with very short delay
            effects.append(Chorus(
                rate_hz=rng.uniform(0.5, 2.0),
                depth=rng.uniform(0.1, 0.3),
                centre_delay_ms=rng.uniform(1.0, 5.0),
            ))
        if effects:
            board = Pedalboard(effects)
            mix_np = board(mix_np, sr)
            tgt_np = board(tgt_np, sr)

    aug_mixture = _to_tensor(mix_np, device)
    aug_target = _to_tensor(tgt_np, device)
    return aug_mixture, aug_target


def create_synthetic_mixture(
    track_pool: list[dict],
    n_stems: int,
    chunk_samples: int,
    sr: int,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Build a synthetic mixture by sampling stems from different tracks.

    Args:
        track_pool:    list of dicts {stem_name: wav_path, ...} per track
        n_stems:       how many stems to combine
        chunk_samples: output length in samples
        sr:            sample rate
        rng:           seeded RNG
    Returns:
        mixture: (1, chunk_samples) summed mixture
        target:  (1, chunk_samples) first stem (arbitrary; caller picks which to use)
        stem_name: name of the first stem
    """
    import soundfile as sf
    if rng is None:
        rng = random.Random()

    # Collect all (stem_name, path) pairs from pool
    all_stems: list[tuple[str, str]] = []
    for track in track_pool:
        for sname, spath in track.items():
            all_stems.append((sname, spath))

    if len(all_stems) < n_stems:
        n_stems = len(all_stems)

    chosen = rng.sample(all_stems, n_stems)
    wavs = []
    for _, path in chosen:
        data, file_sr = sf.read(path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if file_sr != sr:
            import torchaudio
            t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
            resamp = torchaudio.transforms.Resample(file_sr, sr)
            data = resamp(t).squeeze(0).numpy()
        # Random crop
        if len(data) >= chunk_samples:
            start = rng.randint(0, len(data) - chunk_samples)
            data = data[start: start + chunk_samples]
        else:
            data = np.pad(data, (0, chunk_samples - len(data)))
        wavs.append(torch.from_numpy(data.astype(np.float32)).unsqueeze(0))

    mixture = sum(wavs)
    stem_name = chosen[0][0]
    target = wavs[0]
    return mixture, target, stem_name
