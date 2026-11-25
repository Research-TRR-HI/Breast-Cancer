# Paper 1 Experiment 2 — Original Pretrained Mammography Models (NLBS Evaluation)

**Evaluated UTC:** `2026-11-24T22:08:41.195556+00:00`  
**Protocol Policy:** Strict **Category C** (threshold-independent AUROC & AUPRC only; zero RSNA threshold calibration; zero arbitrary `0.5` cutoff; zero target-label tuning).  
**Cluster Bootstrap:** `2000` paired cluster-bootstrap replicates (`seed=42`, resampled by `split_cluster_id`).

---

## 1. Common 4-View NLBS Cohort (All 4 Finalized Models — N = 6,184, 5,868 clusters, 151 cancers)

| Model ID | Source & License | AUROC (95% CI) | AUPRC (95% CI) | Pos Median [p05, p95] | Neg Median [p05, p95] |
|---|---|---|---|---|---|
| `gmic_pretrained` | AGPL-3.0 | **0.8541** [0.8187, 0.8874] | **0.3437** [0.2660, 0.4240] | `0.4878` [0.0381, 0.9921] | `0.0628` [0.0107, 0.4032] |
| `nyu_fourview_pretrained` | AGPL-3.0 | **0.7925** [0.7551, 0.8280] | **0.1490** [0.1066, 0.2118] | `0.1573` [0.0453, 0.8289] | `0.0614` [0.0251, 0.2774] |
| `glam_pretrained` | NYU Non-Commercial Research-Only (LICENSE) / AGPL-3.0 (README conflict) | **0.8006** [0.7613, 0.8371] | **0.1163** [0.0891, 0.1608] | `0.1905` [0.0266, 0.5693] | `0.0503` [0.0132, 0.2500] |
| `end2end_cbis_ddsm` | CC BY-NC-SA 4.0 | **0.5292** [0.4796, 0.5783] | **0.0446** [0.0295, 0.0777] | `0.0163` [0.0065, 0.5903] | `0.0155` [0.0062, 0.1221] |

### Paired Contrasts (Common 4-View NLBS)

| Contrast | Delta AUROC (95% CI) | Delta AUPRC (95% CI) |
|---|---|---|
| `gmic_pretrained_minus_nyu_fourview_pretrained` | +0.0617 [+0.0264, +0.0971] | +0.1947 [+0.1200, +0.2612] |
| `gmic_pretrained_minus_glam_pretrained` | +0.0535 [+0.0257, +0.0801] | +0.2274 [+0.1605, +0.2856] |
| `gmic_pretrained_minus_end2end_cbis_ddsm` | +0.3249 [+0.2653, +0.3823] | +0.2991 [+0.2202, +0.3707] |
| `nyu_fourview_pretrained_minus_glam_pretrained` | -0.0082 [-0.0386, +0.0229] | +0.0327 [-0.0042, +0.0755] |
| `nyu_fourview_pretrained_minus_end2end_cbis_ddsm` | +0.2633 [+0.2077, +0.3200] | +0.1044 [+0.0548, +0.1604] |
| `glam_pretrained_minus_end2end_cbis_ddsm` | +0.2714 [+0.2146, +0.3291] | +0.0717 [+0.0365, +0.1111] |

---

## 2. Full NLBS Cohort (3 View-Agnostic Models — N = 6,359, 5,942 clusters, 151 cancers)

*Note: `nyu_fourview_pretrained` is excluded from Full NLBS because 175 non-4-view exams lack all 4 standard views (no view fabrication).* 

| Model ID | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| `gmic_pretrained` | **0.8554** [0.8205, 0.8883] | **0.3426** [0.2680, 0.4228] |
| `glam_pretrained` | **0.8019** [0.7635, 0.8383] | **0.1139** [0.0859, 0.1581] |
| `end2end_cbis_ddsm` | **0.5286** [0.4782, 0.5777] | **0.0433** [0.0284, 0.0769] |

---

## 3. Cluster-Collapsed Common 4-View Sensitivity (N = 5,868 clusters, 149 positive clusters)

| Model ID | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| `gmic_pretrained` | 0.8505 [0.8141, 0.8844] | 0.3447 [0.2667, 0.4252] |
| `nyu_fourview_pretrained` | 0.7890 [0.7524, 0.8242] | 0.1494 [0.1063, 0.2116] |
| `glam_pretrained` | 0.7953 [0.7556, 0.8325] | 0.1183 [0.0899, 0.1641] |
| `end2end_cbis_ddsm` | 0.5312 [0.4826, 0.5792] | 0.0459 [0.0305, 0.0794] |

---

## 4. Cryptographic SHA-256 Locks & Excluded Candidates

| Model ID | Checkpoint SHA-256 | Case Scores CSV SHA-256 | Lock JSON SHA-256 |
|---|---|---|---|
| `gmic_pretrained` | `3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991` | `94521810b9a78e7047a39d71bc54618ea2c7e95600a6f4fba717b79fdb3a6b36` | `4c84005c751fe321e13262e2d2eea4d73d969e527e6cf458183cbe44d02cbc44` |
| `nyu_fourview_pretrained` | `9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014` | `03e19975edb7a02d297d5b15b672169dd449a86c850488f3f543c5ceddede317` | `aefa55f02dc42639acb2c2ad0b47adc4d7179e6cd1512b831f9f39f535866d3b` |
| `glam_pretrained` | `a647ba9176b3e11d2dec7252054504f33856bcabb2bbe763432d3d20e7c65aec` | `12d2ac770c19a8a646276caa7330d7a7f718d69df92831cfb3648de3dd471885` | `c0ad158f5aa643007d8be9b718163f824e7e225981c2122a40711cb903edb09b` |
| `end2end_cbis_ddsm` | `1c93c4c3ff7f997cc235140340181995d0219162cecd0bf495381b777a2389be` | `40b30414850199bbcab0419f11e8a758301a12b5743e341c7087a5e0b6e679e3` | `6fc816fe53ef3dca235659d5d74367b9c4d6616c6b6cdcf2cc061eea6affdcc4` |

- **`petrini_twoview_efnb3` (`blocked_upstream`):** BLOCKED_UPSTREAM: Both HuggingFace and Google Drive checkpoints (2024-10-01 and 2024-07-01 .pt) are forensically corrupted — truncated at 88.0 MiB (nz_end=92,274,688) with ~405 MiB zero-padding and no ZIP EOCD (PK\x05\x06); torch.load raises BadZipFile.
- **`frcnn_cad` (`disqualified`):** DISQUALIFIED: Official weight URL http://dkrib.web.elte.hu/cad_faster_rcnn/vgg16_frcnn_cad.caffemodel redirects to error.elte.hu/error/452.html (dead, no mirror), CUDA 8.0/Caffe is incompatible with NVIDIA L40S (sm_89), and lesion detection outputs are not natively calibrated screening cancer scores.
- **`mammoscreen_r1` (`rejected`):** REJECTED FOR EXP 2: Fine-tuned on RSNA Screening Mammography (80/10/10 Split 1), violating Experiment 2's strict zero-RSNA-supervision separation from Experiment 1.
- **`petrini_vindr_r2` (`rejected`):** REJECTED FOR EXP 2: Trained on radiological BI-RADS >=3 ('Abnormal') labels rather than pathology/follow-up confirmed breast cancer.
- **`mammo_clip_zeroshot` (`rejected`):** REJECTED FOR EXP 2: Contrastive vision-language foundation model, not a supervised breast cancer screening classifier (already benchmarked with RSNA heads as ce_b5/trust_b5 in Exp 1).
