# IMDB (India) — Six-Model Parallel Execution Summary

**Date:** `2026-11-26`  
**Modal Profile:** `tigerwood697` (`rsna-mammography-vol`, Volume v1)  
**Cohort:** `datasets/exp3_meta/imdb_ffdm_4view_inference_manifest.csv` (`1,665` 4-view FFDM exams / `6,660` images: `304` `r1` + `1,361` `r2`)  
**Status:** **COMPLETE — ALL 6 MODELS LOCKED (`n_exams = 1,665`, `0` crop fallbacks, `0` running Modal apps)**

---

## 1. Checkpoint & Prediction Provenance Locks

All six checkpoints were SHA-256 verified prior to inference, and all six prediction bundles were SHA-256 locked prior to joining Indian cancer outcomes:

| Model ID | Checkpoint SHA-256 | Predictions SHA-256 | Frozen RSNA `98%` / `95%` | Exams (`Fallbacks`) | Completed App ID |
|---|---|---|---|---:|---|
| `ce_b5` | `adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22` | `6b5b442a3b68a808b66d656c01b04114cb4676fca37e1e8d8f929f677d438a76` | `0.096193` / `0.124415` | `1,665` (`0`) | `ap-S3cas1duBShS2EkvWWPMi7` |
| `trust_b5` | `63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb` | `c3a85349e55dc0065e21f66bcdf34d13473ea149ea001382e68be8e0cc790bf0` | `0.179548` / `0.195463` | `1,665` (`0`) | `ap-FVrYxS1rmMLyG1SjjWM8zL` |
| `dinov2_rsna` | `b141c2631ca8e54e5d233239823aafaf18fa3f5d8c62c755339888f41f67d5f5` | `1253742e6e2fac0f6925383b6b803d690930c256f747c060278a92b8bf8ad371` | `0.183436` / `0.206516` | `1,665` (`0`) | `ap-2OXpNGcnsgjlh6KgnnVUoE` |
| `resnet50_rsna` | `ed620c61e38ce47ceac99d73fb1b02c2dae4946552e4469f5395f19b2a0c9d32` | `29de60c599e2247a5e2f87d1b51890def661937d784cfdcfcc5ded3f01714635` | `0.000993` / `0.006931` | `1,665` (`0`) | `ap-SLW92Bbj6PHCDt1PMT5Bv4` |
| `gmic_rsna_finetuned` | `11efb7c65f773fc7f4f7a7bf8b0620c276c63536a3e7e7b2bf84eed6024146d5` | `7544141f7fc5a6b96e1e2c968c7f67cc638cda13c90ec22987cd998d9ab79a03` | `0.130208` / `0.158398` | `1,665` (`0`) | `ap-stg0OBgR14ZIaS2k1QkU2m` |
| `nyu_fourview_rsna_finetuned` | `b42a8b7d623210cd88369b402981b4d1050ea72052851c656c0bbd380532b00e` | `3c1efe8d9ec782b379be19bc6e2c18b1d1d6010c016d59d74c3854e4c642b5ec` | `0.047732` / `0.069086` | `1,665` (`0`) | `ap-tzqZ9twFazHN9ddfKwJrG0` + `ap-Q02Ln1MYavZmUtCzHi2XmG` |

---

## 2. Actual Runtime & Modal Expenditure (`tigerwood697`)

- **Completed 4-Shard L40S Runtimes & Estimated GPU Cost:**
  - `ce_b5` (`bs=32`): `640.90s` (`$0.4205`)
  - `trust_b5` (`bs=32`): `498.80s` (`$0.3272`)
  - `dinov2_rsna` (`bs=32`): `555.33s` (`$0.3643`)
  - `resnet50_rsna` (`bs=32`): `312.00s` (`$0.2047`)
  - `gmic_rsna_finetuned` (`bs=16`, native `2944x1920`): `1,842.88s` (`$1.2090`)
  - `nyu_fourview_rsna_finetuned` (`bs=4` exams, 10-TTA): `~3,218s` (`$2.1114` across 4 shards, resumed cleanly at `$0.00` lock)
- **Failures & Bottleneck Root Cause:**
  - Initial 1-shard concurrent launches (`ap-C1q7tZ...`, `ap-iC31h6...`, `ap-MI0nyJ...`, `ap-cYyErO...`, `ap-jHV4Nu...`, `ap-Udx5AR...`) hit per-shard `--max-spend-usd` caps because Modal Volume v1 FUSE read throughput on `6,660` PNGs / DICOMs in a single container stalled.
  - Adding `ThreadPoolExecutor(max_workers=16)` for cached PNG reads and switching to `--num-shards 4` resolved all FUSE bottlenecks cleanly.
- **Actual Modal Billing (`modal billing summary --profile tigerwood697`):**
  - Baseline Ephemeral Apps (before expansion): `$9.94`
  - Final Ephemeral Apps (after expansion): **`$23.46`** (**`+$13.52` expansion spend**, exceeding the `$5.50` planning target after user-authorized 4-shard completion, while staying **`$0.00` billed** with **`$6.54` free monthly Starter credits remaining** on `tigerwood697`).
  - Volumes: `$5.45` (covered by Free Storage `-$5.45`), Network Egress: `$0.12` (covered by Allowance `-$0.12`), **Billed Cost: `$0.00`**.
  - Active Apps: `0`.
