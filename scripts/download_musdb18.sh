#!/usr/bin/env bash
# Downloads MUSDB18-HQ via the musdb Python package helper.
# MUSDB18-HQ is used ONLY for evaluation — never for training.
set -euo pipefail

DEST="./datasets/musdb18hq"
mkdir -p "$DEST"

python - <<'EOF'
import musdb, sys, os
print("Downloading MUSDB18-HQ (this may take a while) ...")
db = musdb.DB(root=os.environ.get("MUSDB_ROOT", "./datasets/musdb18hq"),
              download=True, is_wav=True)
print(f"Done. {len(db.tracks)} tracks available.")
EOF

echo "MUSDB18-HQ ready at $DEST"
echo "Run: python scripts/prepare_musdb.py --root $DEST"
