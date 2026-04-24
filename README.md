# Query-Conditioned Stem Separation via Latent Diffusion

A text- or audio-query-conditioned audio stem separation system built on latent diffusion.
The user provides an MP3 mixture and a text query (e.g., "isolate the acoustic guitar")
or reference audio, and the system outputs the separated stem.

## Architecture

- **Denoiser**: UNet operating in AudioLDM2's VAE latent space
- **Conditioning**: LAION-CLAP query embeddings via FiLM layers + cross-attention to the mixture latent
- **Encoders**: AudioLDM2 VAE (frozen), LAION-CLAP (frozen)
- **Vocoder**: HiFi-GAN (frozen, matched to AudioLDM2 mel config)

## System Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 12.x
- Single GPU with ~24 GB VRAM (A100 / RTX 4090 class)
- **System library for pyrubberband**:
  - Linux: `sudo apt install rubberband-cli`
  - macOS: `brew install rubberband`

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

### MoisesDB (training)
MoisesDB requires a manual form submission — see `scripts/download_moisesdb.sh` for instructions.
After downloading, run:
```bash
python scripts/prepare_moisesdb.py --root ./datasets/moisesdb
```

### MUSDB18-HQ (evaluation only)
```bash
bash scripts/download_musdb18.sh
python scripts/prepare_musdb.py --root ./datasets/musdb18hq
```

> **Note**: MUSDB18-HQ is used **only** for evaluation and must never appear in training data.

## Training

Three training modes are available:

```bash
# Mode A: positive + negative triplets only
python training/train.py --config configs/default.yaml --mode a

# Mode B: all four triplet types (positive, negative, multi-stem, remove-framing)
python training/train.py --config configs/default.yaml --mode b

# Mode C: fine-tune from checkpoint with consistency + smoothness losses
python training/train.py --config configs/default.yaml --mode c \
    --init_checkpoint ./checkpoints/best_model.pt
```

## Inference

```bash
python inference/separate.py \
    --mixture path/to/mix.mp3 \
    --query "isolate the acoustic guitar" \
    --checkpoint ./checkpoints/best_model.pt \
    --output ./outputs/stem.wav
```

Reference audio query:
```bash
python inference/separate.py \
    --mixture path/to/mix.mp3 \
    --ref_audio path/to/guitar_ref.wav \
    --checkpoint ./checkpoints/best_model.pt \
    --output ./outputs/stem.wav
```

## Evaluation

```bash
python evaluation/metrics.py \
    --musdb_root ./datasets/musdb18hq \
    --checkpoint ./checkpoints/best_model.pt
```

## Project Structure

```
stem_sep/
├── configs/default.yaml        # All hyperparameters
├── data/
│   ├── dataset.py              # MoisesDB loading + triplet construction
│   ├── augmentation.py         # Pitch, time, effects, synthetic mixing
│   └── preprocessing.py        # Chunking, mel spectrogram, VAE encoding
├── models/
│   ├── unet.py                 # Main UNet denoiser
│   ├── conv_blocks.py          # ConvBlock, FiLM, TimestepEmbedding
│   ├── attention.py            # BottleneckAttention (self + cross)
│   └── diffusion.py            # Noise schedule, DDIM sampler, loss functions
├── frozen/
│   ├── vae_wrapper.py          # AudioLDM2 VAE encode/decode interface
│   ├── clap_wrapper.py         # LAION-CLAP text/audio embedding interface
│   └── vocoder.py              # HiFi-GAN mel-to-wav
├── training/
│   ├── train.py                # Main training loop
│   ├── losses.py               # Diffusion MSE, smoothness, consistency
│   └── checkpoint.py           # Save/load utilities
├── inference/
│   ├── separate.py             # End-to-end inference pipeline
│   └── postprocess.py          # Wiener filter, temporal smoothing, OLA
├── evaluation/
│   └── metrics.py              # SDR, SIR, SAR via mir_eval
└── scripts/
    ├── download_moisesdb.sh
    ├── download_musdb18.sh
    ├── prepare_moisesdb.py
    └── prepare_musdb.py
```
