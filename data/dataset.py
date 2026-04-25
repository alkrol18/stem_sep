"""MoisesDB dataset loader with triplet construction."""
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_triplet, create_synthetic_mixture

# --- Query templates ---
_SINGLE_TEMPLATES = [
    "{}",
    "isolate the {}",
    "extract the {}",
    "just the {}",
    "solo {}",
    "remove everything except the {}",
]
_MULTI_TEMPLATES = [
    "isolate the {} and {}",
    "extract the {} and {}",
    "just the {} and {}",
    "solo {} and {}",
    "remove everything except the {} and {}",
]
_REMOVE_TEMPLATES = [
    "remove the {}",
    "mute the {}",
    "everything except the {}",
    "suppress the {}",
]


def _load_index(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def _load_wav_chunk(
    path: str,
    chunk_samples: int,
    sr_target: int,
    rng: random.Random,
) -> np.ndarray:
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if file_sr != sr_target:
        import torchaudio
        t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        data = (
            torchaudio.functional.resample(t, file_sr, sr_target)
            .squeeze(0)
            .numpy()
        )
    if len(data) >= chunk_samples:
        start = rng.randint(0, len(data) - chunk_samples)
        data = data[start: start + chunk_samples]
    else:
        data = np.pad(data, (0, chunk_samples - len(data)))
    return data.astype(np.float32)


class MoisesDataset(Dataset):
    def __init__(self, index_path: str, config: dict, augment: bool = True):
        """
        Args:
            index_path: path to JSON produced by prepare_moisesdb.py
            config:     full config dict (audio + augmentation + triplets keys)
            augment:    whether to apply augmentation
        """
        self.tracks = _load_index(index_path)
        self.audio_cfg = config["audio"]
        self.aug_cfg = config["augmentation"]
        self.triplet_cfg = config["triplets"]
        self.augment = augment
        self.chunk_samples = self.audio_cfg["chunk_samples"]
        self.sr = self.audio_cfg["sample_rate"]

        # Build global stem vocabulary
        vocab: set[str] = set()
        for track in self.tracks:
            for stem in track["stems"]:
                vocab.add(stem["name"])
        self.stem_vocab: list[str] = sorted(vocab)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(idx + random.randint(0, 2**31))
        track = self.tracks[idx]
        stems = track["stems"]          # list of {name, path}
        stem_names = [s["name"] for s in stems]

        # Decide triplet type
        p_neg = self.triplet_cfg["negative_prob"]
        p_multi = self.triplet_cfg["multi_stem_prob"]
        p_remove = self.triplet_cfg["remove_framing_prob"]
        roll = rng.random()

        if roll < p_neg:
            triplet_type = "negative"
        elif roll < p_neg + p_multi:
            triplet_type = "multi"
        elif roll < p_neg + p_multi + p_remove:
            triplet_type = "remove"
        else:
            triplet_type = "positive"

        if triplet_type == "negative":
            # Pick instrument absent from this track
            absent = [s for s in self.stem_vocab if s not in stem_names]
            if not absent:
                triplet_type = "positive"   # fallback
            else:
                absent_name = rng.choice(absent)
                query = rng.choice(_SINGLE_TEMPLATES).format(absent_name)
                mixture_wav = self._load_mixture(track, rng)
                target_wav = np.zeros(self.chunk_samples, dtype=np.float32)
                return self._make_item(mixture_wav, target_wav, query, rng, absent_name)

        if triplet_type == "positive":
            stem = rng.choice(stems)
            stem_name = stem["name"]
            target_wav = _load_wav_chunk(stem["path"], self.chunk_samples, self.sr, rng)
            mixture_wav = self._load_mixture(track, rng, anchor_path=stem["path"],
                                              anchor_wav=target_wav)
            query = rng.choice(_SINGLE_TEMPLATES).format(stem_name)
            return self._make_item(mixture_wav, target_wav, query, rng, stem_name)

        if triplet_type == "multi":
            k = min(rng.randint(2, 3), len(stems))
            chosen = rng.sample(stems, k)
            names = [s["name"] for s in chosen]
            wavs = [_load_wav_chunk(s["path"], self.chunk_samples, self.sr, rng)
                    for s in chosen]
            target_wav = sum(wavs)
            mixture_wav = self._load_mixture(track, rng)
            if k == 2:
                query = rng.choice(_MULTI_TEMPLATES).format(*names)
            else:
                query = "isolate the " + ", ".join(names[:-1]) + " and " + names[-1]
            return self._make_item(mixture_wav, target_wav, query, rng, names[0])

        # remove framing
        excluded = rng.choice(stems)
        excl_name = excluded["name"]
        remaining = [s for s in stems if s["name"] != excl_name]
        if not remaining:
            remaining = stems
        wavs = [_load_wav_chunk(s["path"], self.chunk_samples, self.sr, rng)
                for s in remaining]
        target_wav = sum(wavs)
        mixture_wav = self._load_mixture(track, rng)
        query = rng.choice(_REMOVE_TEMPLATES).format(excl_name)
        return self._make_item(mixture_wav, target_wav, query, rng, excl_name)

    def _load_mixture(
        self,
        track: dict,
        rng: random.Random,
        anchor_path: str | None = None,
        anchor_wav: np.ndarray | None = None,
    ) -> np.ndarray:
        """Load (or reconstruct) the mixture as sum of all stems."""
        if "mixture_path" in track and track["mixture_path"]:
            return _load_wav_chunk(track["mixture_path"], self.chunk_samples, self.sr, rng)
        # Reconstruct from stems
        wavs = []
        for stem in track["stems"]:
            if anchor_path and stem["path"] == anchor_path and anchor_wav is not None:
                wavs.append(anchor_wav)
            else:
                wavs.append(_load_wav_chunk(stem["path"], self.chunk_samples, self.sr, rng))
        return sum(wavs)

    def _make_item(
        self,
        mixture_np: np.ndarray,
        target_np: np.ndarray,
        query: str,
        rng: random.Random,
        stem_name: str,
    ) -> dict[str, Any]:
        mix_t = torch.from_numpy(mixture_np).unsqueeze(0)   # (1, T)
        tgt_t = torch.from_numpy(target_np).unsqueeze(0)    # (1, T)

        if self.augment:
            mix_t, tgt_t = augment_triplet(
                mix_t, tgt_t, stem_name, self.sr, self.aug_cfg, rng=rng
            )

        return {
            "mixture_wav": mix_t,       # (1, chunk_samples)
            "target_stem_wav": tgt_t,   # (1, chunk_samples)
            "query_text": query,
        }
