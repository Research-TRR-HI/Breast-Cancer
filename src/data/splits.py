"""Patient-Level Split Generator for RSNA Screening Mammography.

Prevents all data leakage:
- Grouping is strictly by `patient_id`.
- Every image, breast, view, and examination from one woman stays in the same split.
- Stratified on patient-level cancer label (positive if any image is cancer-positive).
- Split proportions:
    - 70% model fitting (train)
    - 10% internal training-calibration (for recalculating tau_t per epoch)
    - 20% outer validation / operating-threshold calibration
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def generate_patient_splits(
    manifest_path: Path = Path("datasets/rsna/manifest.csv"),
    output_path: Path = Path("datasets/rsna/splits_seed42.csv"),
    seed: int = 42,
    train_frac: float = 0.70,
    calib_frac: float = 0.10,
    test_frac: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Generate patient-level train, train-calib, and val-calib splits.

    Args:
        manifest_path: Path to RSNA manifest CSV.
        output_path: Path to write the resulting splits CSV.
        seed: Random seed for reproducibility.
        train_frac: Fraction of patients for model fitting.
        calib_frac: Fraction of patients for training-time threshold calibration.
        test_frac: Fraction of patients for final threshold validation.

    Returns:
        Tuple of (split_dataframe, split_statistics_dict).
    """
    assert abs(train_frac + calib_frac + test_frac - 1.0) < 1e-5, "Fractions must sum to 1.0"

    df = pd.read_csv(manifest_path)
    rng = np.random.RandomState(seed)

    # Compute patient-level cancer status
    patient_cancer = df.groupby("patient_id")["cancer"].max().reset_index()

    pos_patients = patient_cancer[patient_cancer["cancer"] == 1]["patient_id"].values
    neg_patients = patient_cancer[patient_cancer["cancer"] == 0]["patient_id"].values

    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def _split_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(arr)
        n_train = int(round(n * train_frac))
        n_calib = int(round(n * calib_frac))
        train = arr[:n_train]
        calib = arr[n_train : n_train + n_calib]
        test = arr[n_train + n_calib :]
        return train, calib, test

    pos_train, pos_calib, pos_test = _split_array(pos_patients)
    neg_train, neg_calib, neg_test = _split_array(neg_patients)

    patient_to_split = {}
    for p in pos_train:
        patient_to_split[p] = "train"
    for p in neg_train:
        patient_to_split[p] = "train"

    for p in pos_calib:
        patient_to_split[p] = "train_calib"
    for p in neg_calib:
        patient_to_split[p] = "train_calib"

    for p in pos_test:
        patient_to_split[p] = "val_calib"
    for p in neg_test:
        patient_to_split[p] = "val_calib"

    df["split"] = df["patient_id"].map(patient_to_split)

    # Verify zero leakage across splits
    train_pts = set(df[df["split"] == "train"]["patient_id"])
    calib_pts = set(df[df["split"] == "train_calib"]["patient_id"])
    test_pts = set(df[df["split"] == "val_calib"]["patient_id"])

    assert len(train_pts & calib_pts) == 0, "Patient leakage between train and train_calib!"
    assert len(train_pts & test_pts) == 0, "Patient leakage between train and val_calib!"
    assert len(calib_pts & test_pts) == 0, "Patient leakage between train_calib and val_calib!"

    stats = {}
    for split_name in ["train", "train_calib", "val_calib"]:
        sub = df[df["split"] == split_name]
        n_pts = sub["patient_id"].nunique()
        pos_pts = sub.groupby("patient_id")["cancer"].max().sum()
        stats[split_name] = {
            "total_images": len(sub),
            "unique_patients": n_pts,
            "cancer_positive_patients": int(pos_pts),
            "cancer_prevalence_pct": round(float(pos_pts) / n_pts * 100, 2),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "patient_id",
        "image_id",
        "split",
        "cancer",
        "laterality",
        "view",
        "site_id",
        "machine_id",
        "rel_path",
    ]
    df[out_cols].to_csv(output_path, index=False)
    return df, stats


if __name__ == "__main__":
    _, summary = generate_patient_splits()
    print("Patient-level splits generated successfully:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
