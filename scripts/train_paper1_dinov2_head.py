"""Paper 1 — CPU Cancer-Head Training, RSNA Evaluation & Label-Free NLBS Locking for DINOv2 ViT-S/14.

Protocol:
    1. Consumes extracted FP32 CLS embeddings produced by `scripts/run_paper1_dinov2_modal.py`.
    2. Trains `DINOv2CancerHead` strictly on RSNA `train` labels (zero access to `val_calib` or NLBS).
    3. Selects best checkpoint epoch strictly on RSNA `train_calib` patient-level AUROC.
    4. Evaluates independent RSNA 2-fold OOF `val_calib` source performance and freezes
       full-`val_calib` `@98%` and `@95%` operating thresholds before scoring NLBS.
    5. Produces label-free NLBS image + StudyInstanceUID predictions before any outcome join.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.rsna_dataset import BalancedBatchSampler
from src.evaluation.stage_c_eval import (
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_sha256_file,
    load_label_free_inference_manifest,
)
from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
from src.paper1.dinov2_model import DINOV2_VITS14_SHA256, DINOv2CancerHead
from src.training.calibration import compute_target_recall_threshold, evaluate_dismissal_at_threshold


def _bootstrap_rsna_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    dis95: np.ndarray,
    dis98: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    aurocs, auprcs, s95, d95, s98, d98 = [], [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        sb = scores[idx]
        aurocs.append(compute_auroc(yb, sb))
        auprcs.append(compute_auprc(yb, sb))
        pos = yb == 1
        s95.append(float(np.mean(~dis95[idx][pos])))
        d95.append(float(np.mean(dis95[idx])))
        s98.append(float(np.mean(~dis98[idx][pos])))
        d98.append(float(np.mean(dis98[idx])))

    def _ci(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "auroc": _ci(aurocs),
        "auprc": _ci(auprcs),
        "target_95_sensitivity": _ci(s95),
        "target_95_dismissal": _ci(d95),
        "target_98_sensitivity": _ci(s98),
        "target_98_dismissal": _ci(d98),
    }


def train_and_lock_dinov2_head(
    rsna_emb_pt: Path,
    nlbs_emb_pt: Optional[Path],
    splits_csv: Path,
    val_calib_2fold_csv: Path,
    nlbs_manifest_csv: Path,
    out_dir: Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 0,
    dropout: float = 0.3,
    seed: int = 42,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Train DINOv2 linear/MLP head on RSNA train, select on train_calib, eval 2-fold OOF val_calib, lock NLBS."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rsna_pack = torch.load(str(rsna_emb_pt), map_location="cpu", weights_only=False)
    if str(rsna_pack.get("backbone_sha256", "")).lower() != DINOV2_VITS14_SHA256:
        raise RuntimeError(f"Unexpected DINOv2 backbone_sha256 in {rsna_emb_pt}")

    emb_map = {str(img_id): i for i, img_id in enumerate(rsna_pack["image_ids"])}
    embs = rsna_pack["embeddings"].float()

    splits_df = pd.read_csv(splits_csv)
    splits_df["image_id"] = splits_df["image_id"].astype(str)
    splits_df = splits_df[splits_df["image_id"].isin(emb_map)].copy()
    splits_df["emb_idx"] = splits_df["image_id"].map(emb_map)

    tr_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    tc_df = splits_df[splits_df["split"] == "train_calib"].reset_index(drop=True)
    vc_df = splits_df[splits_df["split"] == "val_calib"].reset_index(drop=True)
    if tr_df.empty or tc_df.empty or vc_df.empty:
        raise RuntimeError("Missing required RSNA train, train_calib, or val_calib rows in embeddings.")

    x_tr = embs[tr_df["emb_idx"].values]
    y_tr = tr_df["cancer"].values.astype(np.float32)
    x_tc = embs[tc_df["emb_idx"].values]
    tc_pids = tc_df["patient_id"].astype(str).tolist()
    tc_lbls = tc_df["cancer"].values.astype(int).tolist()

    head = DINOv2CancerHead(embed_dim=int(embs.shape[1]), hidden_dim=hidden_dim, dropout=dropout)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    sampler = BalancedBatchSampler(
        labels=tr_df["cancer"].values,
        batch_size=batch_size,
        n_pos=max(8, batch_size // 4),
        seed=seed,
        epoch_policy="paper_independent",
        steps_per_epoch=max(20, min(400, len(tr_df) // batch_size)),
    )

    best_auroc = -1.0
    best_epoch = 0
    best_state = None
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        for indices in sampler:
            bx = x_tr[indices]
            by = torch.from_numpy(y_tr[indices]).unsqueeze(1)
            opt.zero_grad()
            loss = crit(head(bx), by)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        sched.step()

        head.eval()
        with torch.no_grad():
            tc_probs = torch.sigmoid(head(x_tc).squeeze(1)).numpy()
        cs = aggregate_case_scores(tc_probs, tc_pids)
        cl = aggregate_case_labels(tc_lbls, tc_pids)
        pids_s = sorted(cs.keys())
        auroc = compute_auroc(np.array([cl[p] for p in pids_s]), np.array([cs[p] for p in pids_s]))
        history.append({"epoch": ep, "loss": round(loss_sum / len(sampler), 5), "train_calib_auroc": round(auroc, 5)})
        if auroc > best_auroc:
            best_auroc = auroc
            best_epoch = ep
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state, strict=True)
    head.eval()
    ckpt_path = out_dir / f"dinov2_vits14_head_seed{seed}.pt"
    torch.save(
        {
            "head_state_dict": best_state,
            "backbone_sha256": DINOV2_VITS14_SHA256,
            "best_epoch": best_epoch,
            "best_train_calib_auroc": float(best_auroc),
            "embed_dim": int(embs.shape[1]),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "seed": seed,
        },
        ckpt_path,
    )
    ckpt_sha = compute_sha256_file(ckpt_path)

    # Independent RSNA val_calib 2-fold cross-fitted evaluation
    x_vc = embs[vc_df["emb_idx"].values]
    with torch.no_grad():
        vc_probs = torch.sigmoid(head(x_vc).squeeze(1)).numpy()
    vc_pids = vc_df["patient_id"].astype(str).tolist()
    vc_lbls = vc_df["cancer"].values.astype(int).tolist()
    vc_cs = aggregate_case_scores(vc_probs, vc_pids)
    vc_cl = aggregate_case_labels(vc_lbls, vc_pids)

    fold_df = pd.read_csv(val_calib_2fold_csv)
    pt_fold = fold_df.groupby("patient_id")["fold"].first().to_dict()
    case_rows = [
        {"patient_id": int(p), "fold": pt_fold[int(p)], "cancer": int(vc_cl[p]), "score": float(vc_cs[p])}
        for p in sorted(vc_cs.keys(), key=int)
    ]
    cdf = pd.DataFrame(case_rows)
    cdf.to_csv(out_dir / "rsna_val_calib_case_predictions.csv", index=False)

    oof_parts = []
    for cal_fold, test_fold in (("fold_a", "fold_b"), ("fold_b", "fold_a")):
        cal = cdf[cdf["fold"] == cal_fold]
        tst = cdf[cdf["fold"] == test_fold].copy()
        t95 = compute_target_recall_threshold(cal[cal["cancer"] == 1]["score"].values, 0.95)
        t98 = compute_target_recall_threshold(cal[cal["cancer"] == 1]["score"].values, 0.98)
        tst["dismissed_95"] = tst["score"] < t95
        tst["dismissed_98"] = tst["score"] < t98
        oof_parts.append(tst)

    oof_df = pd.concat(oof_parts, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    pos_full = cdf[cdf["cancer"] == 1]["score"].values
    tau95_full = float(compute_target_recall_threshold(pos_full, 0.95))
    tau98_full = float(compute_target_recall_threshold(pos_full, 0.98))

    eval_summary: Dict[str, Any] = {
        "model_id": "dinov2_vits14_rsna",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": best_epoch,
        "best_train_calib_auroc": float(best_auroc),
        "rsna_val_calib_auroc": float(compute_auroc(oof_df["cancer"].values, oof_df["score"].values)),
        "rsna_val_calib_auprc": float(compute_auprc(oof_df["cancer"].values, oof_df["score"].values)),
        "rsna_oof_2fold_boot_95ci": _bootstrap_rsna_ci(
            oof_df["score"].values,
            oof_df["cancer"].values,
            oof_df["dismissed_95"].values,
            oof_df["dismissed_98"].values,
            n_boot=n_boot,
            seed=seed,
        ),
        "full_val_calib_thresholds_for_nlbs": {"target_95": tau95_full, "target_98": tau98_full},
    }

    if nlbs_emb_pt is not None and nlbs_emb_pt.exists():
        nlbs_pack = torch.load(str(nlbs_emb_pt), map_location="cpu", weights_only=False)
        if str(nlbs_pack.get("backbone_sha256", "")).lower() != DINOV2_VITS14_SHA256:
            raise RuntimeError(f"Unexpected DINOv2 backbone_sha256 in {nlbs_emb_pt}")
        nlbs_manifest = load_label_free_inference_manifest(nlbs_manifest_csv)
        assert_no_forbidden_columns(nlbs_manifest.columns, str(nlbs_manifest_csv))
        n_map = {str(uid): i for i, uid in enumerate(nlbs_pack["image_ids"])}
        nlbs_manifest["image_uid"] = nlbs_manifest["image_uid"].astype(str)
        missing_nlbs = set(nlbs_manifest["image_uid"]) - set(n_map.keys())
        if missing_nlbs:
            raise RuntimeError(f"Missing {len(missing_nlbs)} NLBS image_uids in {nlbs_emb_pt}")
        x_nlbs = nlbs_pack["embeddings"][nlbs_manifest["image_uid"].map(n_map).values].float()
        with torch.no_grad():
            n_probs = torch.sigmoid(head(x_nlbs).squeeze(1)).numpy()
        nlbs_manifest["prob_cancer"] = n_probs
        exam_scores = aggregate_case_scores(n_probs, nlbs_manifest["StudyInstanceUID"].astype(str).tolist())
        locked_df = pd.DataFrame(
            [
                {
                    "StudyInstanceUID": uid,
                    "score": float(sc),
                    "dismissed_95": bool(sc < tau95_full),
                    "dismissed_98": bool(sc < tau98_full),
                    "threshold_95": tau95_full,
                    "threshold_98": tau98_full,
                }
                for uid, sc in sorted(exam_scores.items())
            ]
        )
        locked_csv = out_dir / "locked_nlbs_predictions.csv"
        locked_df.to_csv(locked_csv, index=False)
        eval_summary["locked_nlbs_predictions_csv"] = str(locked_csv)
        eval_summary["locked_nlbs_sha256"] = compute_sha256_file(locked_csv)
        eval_summary["locked_nlbs_studies"] = int(len(locked_df))

    summary_json = out_dir / "dinov2_head_summary.json"
    summary_json.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Evaluate Paper 1 DINOv2 ViT-S/14 Cancer Head on CPU.")
    parser.add_argument("--rsna-emb", type=Path, required=True)
    parser.add_argument("--nlbs-emb", type=Path, default=None)
    parser.add_argument("--splits-csv", type=Path, default=Path("datasets/rsna/splits_seed42.csv"))
    parser.add_argument("--val-calib-2fold", type=Path, default=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    parser.add_argument("--nlbs-manifest", type=Path, default=Path("datasets/nlbs/inference_manifest_stage_c.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1/dinov2_head"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    res = train_and_lock_dinov2_head(
        rsna_emb_pt=args.rsna_emb,
        nlbs_emb_pt=args.nlbs_emb,
        splits_csv=args.splits_csv,
        val_calib_2fold_csv=args.val_calib_2fold,
        nlbs_manifest_csv=args.nlbs_manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
