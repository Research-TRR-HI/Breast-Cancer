"""Convert End2End CBIS-DDSM Keras .h5 checkpoint to TorchScript and verify <1e-5 parity."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import modal

APP_NAME = "paper1-exp2-end2end-convert"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow-cpu==2.17.0",
        "tf-keras==2.17.0",
        "tf2onnx==1.16.1",
        "onnx==1.17.0",
        "onnx2torch==1.5.15",
        "torch==2.5.1",
        "numpy==1.26.4",
        "h5py==3.13.0",
    )
)


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def convert_and_verify_end2end_cpu() -> Dict[str, Any]:
    import numpy as np
    import onnx
    import tensorflow as tf
    import tf_keras
    import tf2onnx
    import torch
    import torch.nn as nn
    from onnx2torch import convert

    h5_path = RSNA_MOUNT / "models" / "exp2" / "ddsm_resnet50_s10_[512-512-1024]x2.h5"
    ts_path = RSNA_MOUNT / "models" / "exp2" / "end2end_cbis_ddsm_ts.pt"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    keras_model = tf_keras.models.load_model(str(h5_path), compile=False)
    input_spec = (tf.TensorSpec((None, 1152, 896, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=input_spec, opset=13)
    pt_nhwc = convert(onnx_model).eval()

    class NCHWWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
            return self.inner(x_nchw.permute(0, 2, 3, 1).contiguous())

    wrapped = NCHWWrapper(pt_nhwc).eval()
    rng = np.random.default_rng(42)
    x_nhwc = rng.normal(0.0, 25.0, size=(2, 1152, 896, 3)).astype(np.float32)
    x_nchw = torch.from_numpy(np.transpose(x_nhwc, (0, 3, 1, 2)).copy())

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, x_nchw[:1])
        traced.save(str(ts_path))
        ts_loaded = torch.jit.load(str(ts_path), map_location="cpu").eval()
        pt_out = ts_loaded(x_nchw).numpy()

    k_out = keras_model.predict(x_nhwc, verbose=0)
    max_abs_diff = float(np.max(np.abs(k_out - pt_out)))
    if max_abs_diff >= 1e-5:
        raise AssertionError(f"End2End Keras->TorchScript parity failed: max_abs_diff={max_abs_diff} >= 1e-5")

    h = hashlib.sha256()
    with open(ts_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    RSNA_VOL.commit()
    return {
        "h5_path": str(h5_path),
        "ts_path": str(ts_path),
        "ts_bytes": ts_path.stat().st_size,
        "ts_sha256": h.hexdigest(),
        "max_abs_diff_vs_keras": max_abs_diff,
        "keras_sample_out": k_out.tolist(),
        "pt_sample_out": pt_out.tolist(),
    }


@app.local_entrypoint()
def main() -> None:
    res = convert_and_verify_end2end_cpu.remote()
    print(json.dumps(res, indent=2))
