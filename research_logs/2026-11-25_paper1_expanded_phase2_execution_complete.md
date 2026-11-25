# [Paper 1 Expanded] Phase 2 Execution Complete — All Three Experiments

**Date:** `2026-11-25`  
**Track:** Paper 1 Expanded (`studies/paper1/expanded_three_experiment_protocol_v1.md`)  
**Status:** **PHASE 2 COMPLETE (`$22.51 <= $30.00` project cap; `$0.00` billed; IMDB pending manual user download)**

---

## 1. Settled Modal Billing & Storage (`0` Running Apps)

| Profile | Experiment | Wave 2+3 Spend | Phase 2 Spend | Combined New Spend | Total Credits Used / `$30` | Remaining Credits | Final Storage |
|---|---|---:|---:|---:|---:|---:|---:|
| `dryousufmozumder` | Exp 1 | `$0.56` | `$11.61` | `$12.17` | `$16.05` | `~$13.95` | `934.51 GiB` (`< 1 TiB`) |
| `tigerwood693` | Exp 2 | `$2.15` | `$2.42` | `$4.57` | `$8.80` | `~$21.20` | `< 1 TiB` |
| `tigerwood697` | Exp 3 | `$0.93` | `$4.84` | `$5.77` | `$10.29` | `~$19.71` | `< 1 TiB` |
| **Combined** | **All 3** | **`$3.64`** | **`$18.87`** | **`$22.51`** | — | **`~$54.86` account credits** | **`$0.00/day`** |

---

## 2. Experiment Results Summary

1. **Experiment 1 (`dryousufmozumder` — Common RSNA Supervision):**
   - `gmic_rsna_finetuned` (SHA-256 `11efb7c65f773fc7f4f7a7bf8b0620c276c63536a3e7e7b2bf84eed6024146d5`): RSNA `val_calib` AUROC `0.8570` (`target_98=0.130208`, `target_95=0.158398`); NLBS 4-view (`N=6,184`, `151` pos) AUROC **`0.8624`**, `@98%` recall **`96.03%`** (`6` missed), dismissal **`33.15%`**.
   - `nyu_fourview_rsna_finetuned` (SHA-256 `b42a8b7d623210cd88369b402981b4d1050ea72052851c656c0bbd380532b00e`): RSNA `val_calib` AUROC `0.8013` (`target_98=0.047732`, `target_95=0.069086`); NLBS 4-view (`N=6,184`, `151` pos) AUROC **`0.7846`**, `@98%` recall **`96.69%`** (`5` missed), dismissal **`14.68%`**.
2. **Experiment 2 (`tigerwood693` — Original Pretrained Models, Category C):**
   - NLBS 4-view (`N=6,184`, `151` pos): `gmic_pretrained` **`0.8541`**, `glam_pretrained` **`0.8006`**, `nyu_fourview_pretrained` **`0.7925`**, `end2end_cbis_ddsm` **`0.5292`**. (`reports/paper1/expanded/exp2/exp2_summary.{md,json}`)
3. **Experiment 3 (`tigerwood697` — Multiple Unseen External Datasets, Isolated):**
   - **NLBS (`$0` reuse, `N=6,184`):** `gmic_rsna_finetuned` `0.8624`, `nyu_fourview_rsna_finetuned` `0.7846`, `ce_b5` `0.7089`, `trust_b5` `0.7040`, `resnet50_rsna` `0.6452`, `dinov2_rsna` `0.6299`.
   - **CMMD (`N=826` 4-view / `N=1,775` full, diagnostic):** 4-view `nyu_fourview_rsna_finetuned` `0.7137`, `gmic_rsna_finetuned` `0.7059` (full `0.7671`), `dinov2_rsna` `0.6567`, `trust_b5` `0.6474`, `ce_b5` `0.6414`, `resnet50_rsna` `0.4741`.
   - **LUMINA (`N=468`, 5 models, NYU hard-blocked, diagnostic):** `gmic_rsna_finetuned` `0.7426`, `trust_b5` `0.6708`, `ce_b5` `0.6693`, `resnet50_rsna` `0.5753`, `dinov2_rsna` `0.5730`. (`reports/paper1/expanded/exp3/eval/exp3_summary.{md,json}`)
