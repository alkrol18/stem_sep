# Zero-Shot Stem Separation via Query-Conditioned Latent Diffusion

A query-conditioned audio source separation system for **zero-shot stem extraction**: the user provides a music mixture and either a text query (e.g., "isolate the acoustic guitar") or a reference audio clip, and the system produces the requested stem — including stems and instrument categories that were never explicitly labeled during training. Conditioning on a free-form query rather than a fixed stem vocabulary is what enables zero-shot generalization.

## Architecture

- **Denoiser**: UNet operating in AudioLDM2's VAE latent space
- **Conditioning**: LAION-CLAP query embeddings via FiLM layers + cross-attention to the mixture latent. CLAP's shared text/audio embedding space is what makes zero-shot queries possible — any natural-language description that lands in CLAP space can be used as a separation target.
- **Encoders**: AudioLDM2 VAE (frozen), LAION-CLAP (frozen)
- **Vocoder**: HiFi-GAN (frozen, matched to AudioLDM2 mel config)

## System Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 12.x
- Single GPU with ~24 GB VRAM (A100 / RTX 4090 / A5000 class minimum; A6000 preferred for full pipeline)
- **System library for pyrubberband**:
  - Linux: `sudo apt install rubberband-cli`
  - macOS: `brew install rubberband`

## Setup

### Local

```bash
pip install -r requirements.txt
```

### Duke DCC

```bash
module load miniconda/23.9.0
conda create -n stem_sep python=3.10 -y
conda activate stem_sep
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

The CUDA 12.1 torch build is required because cluster GPU nodes have CUDA 12.8 drivers; newer torch builds compiled for CUDA 13 fail at runtime with a "driver too old" error.

### Frozen models

```bash
python scripts/download_audioldm2.py
python scripts/download_clap.py
```

AudioLDM2 caches to `~/.cache/huggingface`. CLAP saves to `checkpoints/clap/`.

## Data Preparation

### MoisesDB (training)

MoisesDB requires a manual form submission at moises.ai (free for non-commercial research). On submission, the download is a single ~83 GB zip with a 7-day expiring URL. Its broad stem vocabulary (11 categories including bowed strings, plucked strings, multiple keyboard types, wind, percussion) is what makes it a strong base for zero-shot generalization beyond the 4-stem vocals/drums/bass/other paradigm.

For DCC, download directly to xtmp to avoid home-directory quota issues:

```bash
mkdir -p /usr/project/xtmp/$USER
wget -O /usr/project/xtmp/$USER/moisesdb.zip "<URL>"
unzip /usr/project/xtmp/$USER/moisesdb.zip -d /usr/project/xtmp/$USER/
python scripts/prepare_moisesdb.py --root /usr/project/xtmp/$USER/moisesdb/moisesdb_v0.1
```

`prepare_moisesdb.py` writes `index.json` listing all 240 tracks and their stems. Stems are stored in subfolders by category (bass, drums, vocals, guitar, piano, etc.) — there are 11 stem types in the released vocabulary.

### MUSDB18-HQ (evaluation only)

```bash
bash scripts/download_musdb18.sh
python scripts/prepare_musdb.py --root ./datasets/musdb18hq
```

> **Note**: MUSDB18-HQ is used **only** for evaluation and must never appear in training data. Performance on its 4 fixed stems is one signal of zero-shot quality, since the model was never trained on MUSDB-specific labels.

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

The triplet construction (especially multi-stem and remove-framing variants in Mode B) is what teaches the model to interpret novel queries at inference rather than memorizing a fixed mapping from stem name to output.

### DCC SLURM submission

```bash
sbatch run_stem_sep.sh
```

The job script requests one GPU on the `compsci-gpu` partition with 4 CPUs, 32 GB RAM, and 72-hour walltime. Use `configs/dcc.yaml` instead of `default.yaml` to point paths at xtmp.

### Monitoring

```bash
squeue -u $USER
tail -f /usr/project/xtmp/$USER/stem_sep_logs/slurm_<JOBID>.out
tensorboard --logdir /usr/project/xtmp/$USER/stem_sep_logs
```

Checkpoints save every 5 epochs by default (`training.checkpoint_every` in config). Eval runs every 10 epochs against MUSDB18 test set if the dataset is present.

## Inference (Zero-Shot)

Text query — works for any natural-language description, including instruments and timbres not seen in MoisesDB's stem vocabulary:
```bash
python inference/separate.py \
    --mixture path/to/mix.mp3 \
    --query "isolate the acoustic guitar" \
    --checkpoint ./checkpoints/best_model.pt \
    --output ./outputs/stem.wav
```

Reference audio query — useful when the target is hard to describe in words (e.g., a specific synth patch or unusual instrument):
```bash
python inference/separate.py \
    --mixture path/to/mix.mp3 \
    --ref_audio path/to/guitar_ref.wav \
    --checkpoint ./checkpoints/best_model.pt \
    --output ./outputs/stem.wav
```

The pipeline chunks input into 5-second segments with 50% overlap, runs DDIM sampling (50 steps by default), decodes via VAE + HiFi-GAN, and stitches with overlap-add. Optional Wiener filtering and temporal smoothing in `postprocess.py`.

## Evaluation

```bash
python evaluation/metrics.py \
    --musdb_root ./datasets/musdb18hq \
    --checkpoint ./checkpoints/best_model.pt
```

Reports SDR, SIR, and SAR on MUSDB18-HQ test split. Because MUSDB18 stems were never seen during training, scores here reflect zero-shot transfer rather than in-distribution performance.

## Project Structure

```
stem_sep/
├── configs/
│   ├── default.yaml            # All hyperparameters
│   └── dcc.yaml                # DCC-specific paths (xtmp)
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
├── scripts/
│   ├── download_audioldm2.py
│   ├── download_clap.py
│   ├── download_moisesdb.sh
│   ├── download_musdb18.sh
│   ├── prepare_moisesdb.py
│   └── prepare_musdb.py
├── run_stem_sep.sh             # SLURM job script
├── requirements.txt
└── README.md
```

## DCC Storage Layout

Home directory has limited quota (~75 GB). Large files live in xtmp:

```
/home/users/$USER/stem_sep/                       code, configs, scripts
/usr/project/xtmp/$USER/moisesdb/                 dataset (~83 GB)
/usr/project/xtmp/$USER/stem_sep_checkpoints/     model checkpoints
/usr/project/xtmp/$USER/stem_sep_logs/            SLURM + tensorboard logs
/usr/project/xtmp/$USER/stem_sep_outputs/         separated audio outputs
```

## References

- AudioLDM 2: Liu et al., 2023. Holistic latent diffusion audio generation.
- LAION-CLAP: Wu et al., 2023. Contrastive language-audio pretraining.
- MoisesDB: Pereira et al., 2023. A dataset for source separation beyond 4 stems.
- DDIM: Song et al., 2020. Denoising diffusion implicit models.
