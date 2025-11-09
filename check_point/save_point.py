import os
import torch
import logging

def save_checkpoint(state, is_best, checkpoint_dir="./checkpoints", filename="last.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)
        logging.info(f"✅ Saved new best model to {best_path}")
    else:
        logging.info(f"💾 Checkpoint saved to {path}")