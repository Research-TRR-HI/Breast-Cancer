"""RSNA PyTorch Dataset, Canonical Balanced Minibatch Sampler, and Deterministic Draw Schedule.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Section 3.9 (Page 9)
    - Minibatch size 80: 20 cancer-positive and 60 cancer-negative images.
    - Sampling rule:
      "each training minibatch was constructed with 20 cancer-positive and 60 cancer-negative
       samples. Samples were drawn independently for each minibatch, allowing minority-class
       examples to be reused across minibatches as needed."
    - Epoch length note:
      Because the original paper states 20 epochs with batch size 80 (20 pos / 60 neg)
      without stating an explicit integer number of minibatches per epoch,
      ``steps_per_epoch`` defaults to ``ceil(N_neg / n_neg)`` (~620 on 70% RSNA train)
      and can be configured explicitly.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.data.preprocessing import (
    AugmentationParams,
    preprocess_dicom,
    sample_augmentation_params,
)


@dataclass(frozen=True)
class BatchDrawRecord:
    """Deterministic occurrence record linking a minibatch slot to its sample and augmentation."""

    epoch: int
    batch_idx: int
    indices: List[int]
    aug_params: List[AugmentationParams]


class RSNADataset(Dataset):
    """PyTorch Dataset for RSNA Screening Mammography images."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        images_root: Union[str, Path],
        target_size: Tuple[int, int] = (1520, 912),
        is_train: bool = False,
        aug_seed: int = 42,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.target_size = target_size
        self.is_train = bool(is_train)
        self.aug_rng = np.random.default_rng(aug_seed)
        self.transform = transform

        self.labels = self.df["cancer"].values.astype(np.int64)
        self.patient_ids = self.df["patient_id"].astype(str).values
        self.image_ids = self.df["image_id"].astype(str).values
        self.rel_paths = self.df["rel_path"].values
        self.lateralities = (
            self.df["laterality"].astype(str).values
            if "laterality" in self.df.columns
            else np.array([""] * len(self.df))
        )

    def __len__(self) -> int:
        return len(self.df)

    def get_item_with_aug(
        self,
        idx: int,
        augment_params: Optional[AugmentationParams] = None,
    ) -> Tuple[torch.Tensor, int, str, str]:
        rel = self.rel_paths[idx]
        full_path = self.images_root / rel
        label = int(self.labels[idx])
        patient_id = self.patient_ids[idx]
        image_id = self.image_ids[idx]
        lat = str(self.lateralities[idx]).strip().upper()
        if lat not in ("L", "R"):
            lat = ""

        aug = augment_params
        if aug is None and self.is_train:
            aug = sample_augmentation_params(self.aug_rng)

        tensor = preprocess_dicom(
            full_path,
            target_size=self.target_size,
            laterality=lat if lat else None,
            augment_params=aug,
        )
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label, patient_id, image_id

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        return self.get_item_with_aug(idx, augment_params=None)


class BalancedBatchSampler(Sampler[List[int]]):
    """Canonical batch sampler enforcing 20 positives + 60 negatives per batch of 80.

    Epoch policies:
        - "paper_independent" (default, TRUST Section 3.9):
          Draws 20 positives and 60 negatives independently for every minibatch
          (without replacement within a single minibatch, allowing reuse across
          minibatches as needed).
        - "full_neg_pass" (documented alternative):
          Iterates through 100% of shuffled negatives once per epoch while cycling
          reshuffled positives with wraparound.
        - "pos_pass" (legacy baseline):
          floor(N_pos / n_pos) batches per epoch.
    """

    SUPPORTED_POLICIES = ("paper_independent", "full_neg_pass", "pos_pass")

    def __init__(
        self,
        labels: Union[List[int], np.ndarray, torch.Tensor],
        batch_size: int = 80,
        n_pos: int = 20,
        seed: int = 42,
        shuffle: bool = True,
        epoch_policy: str = "paper_independent",
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        if epoch_policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown epoch_policy: {epoch_policy!r}. Expected one of {self.SUPPORTED_POLICIES}."
            )
        labels_arr = np.asarray(labels).flatten()
        self.pos_indices = np.where(labels_arr == 1)[0]
        self.neg_indices = np.where(labels_arr == 0)[0]
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("BalancedBatchSampler requires at least one positive and one negative sample.")

        self.batch_size = int(batch_size)
        self.n_pos = min(int(n_pos), self.batch_size)
        self.n_neg = self.batch_size - self.n_pos
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch_policy = epoch_policy
        self.rng = np.random.RandomState(self.seed)

        if steps_per_epoch is not None:
            self.n_batches = max(int(steps_per_epoch), 1)
        elif epoch_policy in ("paper_independent", "full_neg_pass"):
            self.n_batches = max(int(math.ceil(len(self.neg_indices) / max(self.n_neg, 1))), 1)
        else:
            self.n_batches = max(len(self.pos_indices) // max(self.n_pos, 1), 1)

    def __len__(self) -> int:
        return self.n_batches

    def get_state(self) -> Dict[str, object]:
        """Serialize sampler RNG state and configuration for exact checkpoint resume."""
        return {
            "seed": self.seed,
            "epoch_policy": self.epoch_policy,
            "batch_size": self.batch_size,
            "n_pos": self.n_pos,
            "n_batches": self.n_batches,
            "rng_state": self.rng.get_state(),
        }

    def set_state(self, state: Dict[str, object]) -> None:
        """Restore sampler RNG state."""
        if state.get("epoch_policy") != self.epoch_policy:
            raise ValueError(
                f"Sampler policy mismatch: expected {self.epoch_policy}, got {state.get('epoch_policy')}"
            )
        self.rng.set_state(state["rng_state"])

    def __iter__(self) -> Iterator[List[int]]:
        if self.epoch_policy == "paper_independent":
            replace_pos = len(self.pos_indices) < self.n_pos
            replace_neg = len(self.neg_indices) < self.n_neg
            for _ in range(self.n_batches):
                pos_draw = self.rng.choice(self.pos_indices, size=self.n_pos, replace=replace_pos)
                neg_draw = self.rng.choice(self.neg_indices, size=self.n_neg, replace=replace_neg)
                batch = [int(i) for i in pos_draw] + [int(i) for i in neg_draw]
                if self.shuffle:
                    self.rng.shuffle(batch)
                yield batch
            return

        pos = self.pos_indices.copy()
        neg = self.neg_indices.copy()
        if self.shuffle:
            self.rng.shuffle(pos)
            self.rng.shuffle(neg)

        pos_ptr = 0
        neg_ptr = 0
        for _ in range(self.n_batches):
            batch: List[int] = []
            for _ in range(self.n_pos):
                if pos_ptr >= len(pos):
                    if self.shuffle:
                        self.rng.shuffle(pos)
                    pos_ptr = 0
                batch.append(int(pos[pos_ptr]))
                pos_ptr += 1

            for _ in range(self.n_neg):
                if neg_ptr >= len(neg):
                    if self.shuffle:
                        self.rng.shuffle(neg)
                    neg_ptr = 0
                batch.append(int(neg[neg_ptr]))
                neg_ptr += 1

            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch


def build_epoch_draw_schedule(
    sampler: BalancedBatchSampler,
    epoch: int,
    aug_rng: np.random.Generator,
) -> List[BatchDrawRecord]:
    """Generate an identical, ordered batch + augmentation draw schedule for online and cached training."""
    schedule: List[BatchDrawRecord] = []
    for batch_idx, indices in enumerate(sampler):
        aug_list = [sample_augmentation_params(aug_rng) for _ in indices]
        schedule.append(
            BatchDrawRecord(
                epoch=int(epoch),
                batch_idx=int(batch_idx),
                indices=[int(i) for i in indices],
                aug_params=aug_list,
            )
        )
    return schedule
