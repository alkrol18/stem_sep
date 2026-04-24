"""
Scans a MoisesDB directory, validates stem files, and writes index.json.

Expected layout after extraction:
  <root>/
    <track_id>/
      mixture.wav         (optional)
      <stem_name>.wav     (one per stem)
      ...

Outputs:
  <root>/index.json  — list of track dicts used by MoisesDataset
"""
import argparse
import json
import os
from pathlib import Path


def scan(root: str) -> list[dict]:
    root_path = Path(root)
    tracks = []
    for track_dir in sorted(root_path.iterdir()):
        if not track_dir.is_dir():
            continue
        wav_files = list(track_dir.glob("*.wav"))
        if not wav_files:
            continue
        mixture_path = track_dir / "mixture.wav"
        stems = []
        for wav in wav_files:
            if wav.name == "mixture.wav":
                continue
            stem_name = wav.stem       # filename without extension
            stems.append({"name": stem_name, "path": str(wav)})
        if not stems:
            continue
        tracks.append({
            "track_id": track_dir.name,
            "mixture_path": str(mixture_path) if mixture_path.exists() else None,
            "stems": stems,
        })
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to MoisesDB root directory")
    args = parser.parse_args()

    tracks = scan(args.root)
    if not tracks:
        print(f"[warn] No tracks found in {args.root}")
        return

    out_path = Path(args.root) / "index.json"
    with open(out_path, "w") as f:
        json.dump(tracks, f, indent=2)

    stems_total = sum(len(t["stems"]) for t in tracks)
    stem_names = sorted({s["name"] for t in tracks for s in t["stems"]})
    print(f"[prepare_moisesdb] {len(tracks)} tracks, {stems_total} stems")
    print(f"[prepare_moisesdb] stem vocabulary ({len(stem_names)}): {stem_names}")
    print(f"[prepare_moisesdb] index saved → {out_path}")


if __name__ == "__main__":
    main()
