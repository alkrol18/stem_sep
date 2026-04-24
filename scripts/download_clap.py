import urllib.request
import os

os.makedirs("checkpoints/clap", exist_ok=True)
dest = "checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt"

if os.path.exists(dest):
    print(f"Already exists: {dest}")
else:
    url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
    print("Downloading CLAP checkpoint (~900MB)...")
    def progress(count, block, total):
        pct = count * block * 100 // total
        print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nDone.")
