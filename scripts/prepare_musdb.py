"""
Scans a MUSDB18-HQ directory and writes index.json.
MUSDB18-HQ is used for evaluation ONLY — never appears in training.

Expected layout:
  <root>/
    train/ or test/
      <track_name>/
        mixture.wav
        vocals.wav
        drums.wav
        bass.wav
        other.wav
"""
import argparse
import json
from pathlib import Path

STEM_NAMES = ["vocals", "drums", "bass", "other"]


def scan(root: str) -> dict[str, list[dict]]:
    root_path = Path(root)
    result = {}
    for split in ["train", "test"]:
        split_dir = root_path / split
        if not split_dir.exists():
            continue
        tracks = []
        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            mixture_path = track_dir / "mixture.wav"
            stems = []
            for sn in STEM_NAMES:
                p = track_dir / f"{sn}.wav"
                if p.exists():
                    stems.append({"name": sn, "path": str(p)})
            if not stems:
                continue
            tracks.append({
                "track_id": track_dir.name,
                "mixture_path": str(mixture_path) if mixture_path.exists() else None,
                "stems": stems,
            })
        result[split] = tracks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    splits = scan(args.root)
    out_path = Path(args.root) / "index.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    for split, tracks in splits.items():
        print(f"[prepare_musdb] {split}: {len(tracks)} tracks")
    print(f"[prepare_musdb] index saved → {out_path}")


if __name__ == "__main__":
    main()
