# [Paper 1] Six-Model External Validation Benchmark & Agent 4 GMIC Fine-Tuning Pilot

**Date:** 2026-11-24  
**Track:** Paper 1 (`Paper1_External_Validation_Research_Context.md`)  
**Status:** Primary 6-Model Benchmark COMPLETE & LOCKED (`reports/paper1/six_model_final/six_model_summary.{md,json}`); Agent 4 GMIC Fine-Tuning Pilot COMPLETE (Awaiting Full-Train Approval)

---

## 1. Pre-Execution Fixes & Unskipped Integration Tests
Merged 4 isolated agent branches (`paper1-nyu`, `paper1-dinov2`, `paper1-resnet50`, `paper1-specialist-finetuning`) into `main`:
- `paper1-nyu`: Fixed Windows CRLF vs LF SHA-256 mismatch on Step-1 CSV manifests (`compute_text_lf_sha256`) and `NLBS_VOL` `create_if_missing=True` for backup workspaces.
- `paper1-dinov2`: Added `backbone_sha256` verification on chunk resume, fixed hardcoded `hasinishrak2015` profile & `modal volume get` flag order, and enforced strict `image_uid` completeness in `train_paper1_dinov2_head.py`.
- `paper1-resnet50`: Fixed fatal batch-0 step-0 2D-vs-3D augmentation crash in `prepare_resnet50_tensor_from_uint16`, added deterministic 2-batch CPU prefetch + NVMe `/tmp` inference staging, and upgraded to 15-epoch training (`train_calib` early selection).
- `paper1-specialist-finetuning`: Implemented malignancy-only RSNA fine-tuning pipelines (`src/paper1/gmic_finetune.py`, `src/paper1/nyu_finetune.py`, `scripts/run_paper1_{gmic,nyu}_finetune_modal.py`), 5/5 CPU tests (`tests/test_paper1_specialist_finetuning.py`), and `studies/paper1/specialist_finetuning_protocol_v1.md`.
- **Integration Tests (`python -m unittest discover -s tests -p "test_paper1_*.py" -v`):** `29` total (`27/27` runnable PASS including all `external/GMIC`, `external/breast_cancer_classifier`, and `external/dinov2` tests; `2` skipped only for uncached local DINOv2/ResNet-50 weights already SHA-256 verified on Modal).

---

## 2. Parallel Execution & Modal Spend Ledger (`$0.00` Billed Across All 5 Accounts)
1. **Agent 1 (`nyu_fourview_pretrained`, `hasinishrak2015` -> `hasinishrak74001`):**
   - RSNA `val_calib` (`2,382` exams / `9,528` imgs, `4` shards, `batch_exams=4`, `10`-TTA, FP32) on `hasinishrak2015`: **`USD 0.9180`**.
   - CPU RSNA Calibration: `tau_98 = 0.021428`, `tau_95 = 0.034991` (`reports/paper1/nyu_rsna_calibration.json`, `sha256: db30c942...`).
   - 96-stream cloud-to-cloud failover (`87.7426 GiB`, `24,752` files) to `hasinishrak74001`: **`USD 0.1174`** (`all_verified: true`).
   - NLBS `nlbs_4view` (`6,184` exams / `24,736` imgs) on `hasinishrak74001`: **`USD 2.3040`** (`sha256: 03e19975...`).
2. **Agent 2 (`dinov2_rsna`, `tigerwood693`):**
   - L40S NVMe feature extraction (`54,706` RSNA in `418.58s` @ `130.69 img/s` + `26,988` NLBS in `207.89s` @ `129.82 img/s`): **`~USD 0.44`**.
   - Local CPU head training: best epoch `2` (`train_calib` AUROC `0.7469`), `6,359` NLBS studies locked (`sha256: c27a8b77...`).
3. **Agent 3 (`resnet50_rsna`, `tigerwood697`):**
   - 15-epoch L40S AMP training (`600` steps/ep, `bs=64`): best epoch `2` (`train_calib` AUROC `0.7127`).
   - L40S NVMe inference (`10,982` RSNA in `39.85s`, `26,988` NLBS in `98.70s`): total `tigerwood697` session spend **`USD 4.11`**.
4. **Agent 4 (`gmic_rsna_finetuned` Prep & Pilot, `dryousufmozumder`):**
   - 96-stream `rsna/raw` transfer (`54,713` files / `293.11 GiB`): **`USD 0.2239`** (`all_verified: true`).
   - Built & verified `38,251` `rsna_train` (`216.924 GiB`) + `5,473` `rsna_train_calib` (`31.119 GiB`) native `(2944, 1920)` GMIC `uint16` PNGs (`0.0` parity diff).
   - GMIC L40S pilot (`16` steps, `batch=8`, `partial` = `10.63M / 14.11M` params): `mean_loss = 1.4435`, `1.078 GB` VRAM, `5.1 img/s` -> **`projected_10ep_usd = USD 6.18`** (`2.615h`). Total Agent 4 new spend: **`USD 3.27`** (`<= USD 3.55` cap).

---

## 3. Primary Six-Model External Validation Results (`N = 6,184` Common 4-View NLBS, `5,868` Clusters, `151` Cancers)

| Stratum | Model | RSNA AUROC | NLBS AUROC | Delta AUROC | `@98%` Sens (Missed) | `@98%` Dismissal | `@95%` Sens (Missed) | `@95%` Dismissal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A (RSNA-Supervised) | `ce_b5` | `0.7428` | `0.7089` | `-0.0339` | `90.07%` (`15/151`) | `30.48%` | `80.13%` (`30/151`) | `49.66%` |
| A (RSNA-Supervised) | `trust_b5` | `0.7442` | `0.7040` | `-0.0403` | `84.77%` (`23/151`) | `39.57%` | `80.79%` (`29/151`) | `49.29%` |
| A (RSNA-Supervised) | `dinov2_rsna` | `0.6302` | `0.6299` | `-0.0002` | `98.01%` (`3/151`) | `9.67%` | `94.04%` (`9/151`) | `17.43%` |
| A (RSNA-Supervised) | `resnet50_rsna` | `0.6580` | `0.6452` | `-0.0128` | `97.35%` (`4/151`) | `5.35%` | `84.77%` (`23/151`) | `19.78%` |
| B (Pretrained-Direct) | `gmic_pretrained` | `0.8301` | `0.8541` | `+0.0240` | `97.35%` (`4/151`) | `24.27%` | `93.38%` (`10/151`) | `38.47%` |
| B (Pretrained-Direct) | `nyu_fourview_pretrained` | `0.7721` | `0.7925` | `+0.0203` | `100.00%` (`0/151`) | `1.94%` | `99.34%` (`1/151`) | `15.60%` |
