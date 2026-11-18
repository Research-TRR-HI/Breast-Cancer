"""Pinned Environment Specifications and Runtime Metadata Collection."""

import platform
import sys
from typing import Any, Dict, Tuple


PINNED_MODAL_PYTHON_VERSION = "3.12"

PINNED_MODAL_PIP_PACKAGES: Tuple[str, ...] = (
    "torch==2.5.1",
    "torchvision==0.20.1",
    "numpy==2.5.3",
    "pandas==2.3.3",
    "scipy==1.17.1",
    "pydicom==3.0.2",
    "opencv-python-headless==4.11.0.86",
    "pillow==12.3.0",
    "omegaconf==2.3.0",
    "pylibjpeg==2.0.0",
    "pylibjpeg-libjpeg==2.1.0",
    "pylibjpeg-openjpeg==2.1.0",
    "python-gdcm==3.0.24",
)


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def collect_runtime_environment_metadata(precision_policy: str = "amp") -> Dict[str, Any]:
    """Capture exact runtime Python, PyTorch, CUDA, GPU, and library versions."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    cuda_version = str(torch.version.cuda) if cuda_available else None
    cudnn_version = int(torch.backends.cudnn.version()) if (cuda_available and torch.backends.cudnn.is_available()) else None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torchvision_version": _safe_version("torchvision"),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "numpy_version": _safe_version("numpy"),
        "scipy_version": _safe_version("scipy"),
        "pandas_version": _safe_version("pandas"),
        "pydicom_version": _safe_version("pydicom"),
        "opencv_version": _safe_version("cv2"),
        "pillow_version": _safe_version("PIL"),
        "omegaconf_version": _safe_version("omegaconf"),
        "precision_policy": str(precision_policy),
        "pinned_modal_python": PINNED_MODAL_PYTHON_VERSION,
        "pinned_direct_packages": list(PINNED_MODAL_PIP_PACKAGES),
    }
