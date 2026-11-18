"""Mammo-CLIP Checkpoint Loader and State-Dict Verifier.

Reference:
    Mammo-CLIP: A Vision Language Foundation Model to Enhance Data
    Efficiency and Robustness in Mammography (MICCAI 2024)
    HuggingFace: shawn24/Mammo-CLIP/Pre-trained-checkpoints/b5-model-best-epoch-7.tar
"""

from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn as nn


def extract_mammo_clip_encoder_state(
    checkpoint_path: Union[str, Path],
    prefix: str = "image_encoder.",
) -> Dict[str, torch.Tensor]:
    """Extract and rename vision backbone weights from Mammo-CLIP tar checkpoint.

    Args:
        checkpoint_path: Path to b5-model-best-epoch-7.tar.
        prefix: Prefix used for encoder parameters in Mammo-CLIP.

    Returns:
        State dictionary compatible with Mammo-CLIP's custom EfficientNet-B5.
    """
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
        # Fallback if checkpoint was saved directly as encoder state
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
) -> Dict[str, Union[int, List[str], bool]]:
    """Load Mammo-CLIP weights into model and report compatibility diagnostics.

    Args:
        model: Target PyTorch model (e.g. EfficientNet or MammoCLIPEncoder).
        checkpoint_path: Path to downloaded checkpoint.
        strict: Whether to enforce strict key matching. Defaults to True.

    Returns:
        Diagnostics dictionary with loaded, missing, and unexpected key counts.
    """
    state_dict = extract_mammo_clip_encoder_state(checkpoint_path)
    target_module = getattr(model, "encoder", model)
    missing, unexpected = target_module.load_state_dict(state_dict, strict=strict)

    target_keys = set(target_module.state_dict().keys())
    matched_keys = set(state_dict.keys()) & target_keys

    diagnostics = {
        "total_checkpoint_keys": len(state_dict),
        "total_target_keys": len(target_keys),
        "matched_keys_count": len(matched_keys),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": list(missing)[:10],
        "unexpected_keys_sample": list(unexpected)[:10],
        "is_fully_compatible": len(missing) == 0 and len(unexpected) == 0,
    }
    return diagnostics
