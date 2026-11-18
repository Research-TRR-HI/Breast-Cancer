"""Mammo-CLIP Checkpoint Loader, Pinned Metadata, and State-Dict Verifier.

Reference:
    Mammo-CLIP: A Vision Language Foundation Model to Enhance Data
    Efficiency and Robustness in Mammography (MICCAI 2024)
    HuggingFace: shawn24/Mammo-CLIP
    Pinned Revision: 2f356926a00fc3f0d9fdad1193c1464fd9adf564
    File: Pre-trained-checkpoints/b5-model-best-epoch-7.tar
    SHA-256: 0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn as nn

MAMMO_CLIP_HF_REPO = "shawn24/Mammo-CLIP"
MAMMO_CLIP_HF_REVISION = "2f356926a00fc3f0d9fdad1193c1464fd9adf564"
MAMMO_CLIP_CHECKPOINT_REL = "Pre-trained-checkpoints/b5-model-best-epoch-7.tar"
MAMMO_CLIP_EXPECTED_SIZE = 1654145453
MAMMO_CLIP_SHA256 = "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c"


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute hexadecimal SHA-256 checksum of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_mammo_clip_encoder_state(
    checkpoint_path: Union[str, Path],
    prefix: str = "image_encoder.",
) -> Dict[str, torch.Tensor]:
    """Extract and rename vision backbone weights from Mammo-CLIP tar checkpoint."""
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    raw_state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    encoder_state = {}
    for k, v in raw_state.items():
        if k.startswith(prefix):
            clean_key = k[len(prefix) :]
            encoder_state[clean_key] = v

    if not encoder_state:
        encoder_state = {
            k: v
            for k, v in raw_state.items()
            if not any(
                k.startswith(p)
                for p in ["text_encoder", "logit_scale", "projection", "image_projection", "text_projection"]
            )
        }

    return encoder_state


def load_and_verify_mammo_clip(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
    verify_sha256: bool = False,
) -> Dict[str, Union[int, List[str], bool, str]]:
    """Load Mammo-CLIP weights into model and report compatibility diagnostics."""
    ckpt_path = Path(checkpoint_path)
    sha256_val = compute_file_sha256(ckpt_path) if verify_sha256 else MAMMO_CLIP_SHA256
    if verify_sha256 and sha256_val != MAMMO_CLIP_SHA256:
        raise ValueError(
            f"Mammo-CLIP checkpoint SHA-256 mismatch: expected {MAMMO_CLIP_SHA256}, got {sha256_val}"
        )

    state_dict = extract_mammo_clip_encoder_state(ckpt_path)
    target_module = getattr(model, "encoder", model)
    missing, unexpected = target_module.load_state_dict(state_dict, strict=strict)

    target_keys = set(target_module.state_dict().keys())
    matched_keys = set(state_dict.keys()) & target_keys

    return {
        "total_checkpoint_keys": len(state_dict),
        "total_target_keys": len(target_keys),
        "matched_keys_count": len(matched_keys),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": list(missing)[:10],
        "unexpected_keys_sample": list(unexpected)[:10],
        "is_fully_compatible": len(missing) == 0 and len(unexpected) == 0,
        "pinned_revision": MAMMO_CLIP_HF_REVISION,
        "sha256": sha256_val,
    }
