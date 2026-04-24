from diffusers import AudioLDM2Pipeline

print("Downloading AudioLDM2 (~3GB, this will cache to ~/.cache/huggingface)...")
AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2")
print("Done.")
