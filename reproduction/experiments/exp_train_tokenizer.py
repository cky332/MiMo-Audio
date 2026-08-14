"""Train the tiny tokenizer on the repo's real speech and cache it for the
segmentation (track 2) and OOD (track 3) experiments."""

import os
import torch

from common import ARTIFACTS_DIR, train_tiny_tokenizer

if __name__ == "__main__":
    tok, cfg, hist = train_tiny_tokenizer(steps=500, batch=4, lr=2e-3, seed=0)
    path = os.path.join(ARTIFACTS_DIR, "tok_realspeech.pt")
    torch.save({"state_dict": tok.state_dict()}, path)
    first = sum(hist["recon"][:20]) / 20
    last = sum(hist["recon"][-20:]) / 20
    print(f"recon loss first20={first:.3f} last20={last:.3f} -> saved {path}")
