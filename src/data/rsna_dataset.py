"""RSNA PyTorch Dataset and Balanced Batch Sampler.

Implements TRUST Section 3.2 minibatch sampling:
- Batch size: 80 (20 cancer-positive images, 60 cancer-negative images).
- Dynamic ratio support for smaller smoke-test batches (e.g. batch size 4 -> 1 pos, 3 neg).
"""

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.data.preprocessing import preprocess_dicom


class RSNADataset(Dataset):
    """PyTorch Dataset for RSNA Screening Mammography images."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        images_root: Union[str, Path],
        target_size: Tuple[int, int] = (1520, 912),
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.target_size = target_size
        self.transform = transform

        self.labels = self.df["cancer"].values.astype(np.int64)
        self.patient_ids = self.df["patient_id"].astype(str).values
        self.image_ids = self.df["image_id"].astype(str).values
        self.rel_paths = self.df["rel_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        rel = self.rel_paths[idx]
        full_path = self.images_root / rel
        label = int(self.labels[idx])
        patient_id = self.patient_ids[idx]
        image_id = self.image_ids[idx]

        tensor = preprocess_dicom(full_path, target_size=self.target_size)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label, patient_id, image_id


class BalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler enforcing fixed positive/negative ratio per batch.

    TRUST default: batch size 80 = 20 positives + 60 negatives.
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray, torch.Tensor],
        batch_size: int = 80,
        n_pos: int = 20,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        labels_arr = np.asarray(labels).flatten()
        self.pos_indices = np.where(labels_arr == 1)[0]
        self.neg_indices = np.where(labels_arr == 0)[0]

        self.batch_size = batch_size
        self.n_pos = min(n_pos, batch_size)
        self.n_neg = batch_size - self.n_pos
        self.seed = seed
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Number of batches determined by exhausting the positive samples
        self.n_batches = max(len(self.pos_indices) // self.n_pos, 1)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        pos = self.pos_indices.copy()
        neg = self.neg_indices.copy()

        if self.shuffle:
            self.rng.shuffle(pos)
            self.rng.shuffle(neg)

        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.n_batches):
            batch = []

            # Sample positives (with wraparound if exhausted)
            for _ in range(self.n_pos):
                if pos_ptr >= len(pos):
                    if self.shuffle:
                        self.rng.shuffle(pos)
                    pos_ptr = 0
                batch.append(int(pos[pos_ptr]))
                pos_ptr += 1

            # Sample negatives (with wraparound if exhausted)
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
