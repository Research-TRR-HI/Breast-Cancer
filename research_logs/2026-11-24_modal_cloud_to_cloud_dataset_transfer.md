# Research Log: Parallel Modal Cloud-to-Cloud Multi-Workspace Dataset Transfer

**Date**: 2026-11-24  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Cloud Infrastructure & Dataset Distribution (`[Shared / Paper 1]`)  
**Milestone Scope**: Zero-local-routing cloud-to-cloud replication of verified Stage-1 caches, NYU 4-view caches, canonical splits, and pretrained checkpoints from `hasinishrak2015` & `hasinishrak74001` to `tigerwood693`, `tigerwood697`, and `dryousufmozumder`  
**Status**: VERIFIED COMPLETE (ZERO TRAINING LAUNCHED)

---

## 1. Research Objective & Context
- Offload pending Paper 1 training and inference compute (`dinov2_rsna`, `resnet50_rsna`, and `dryousufmozumder` Option B including `nyu_fourview_pretrained` & Mammo-CLIP `b5`) from low-credit `hasinishrak2015` (`USD 7.50` credits remaining) onto three dedicated destination Modal accounts:
  - `tigerwood693` — **DINOv2 bundle** (`101.9755 GiB` / `81,731` files)
  - `tigerwood697` — **ResNet-50 bundle** (`101.9888 GiB` / `81,731` files)
  - `dryousufmozumder` — **Option B bundle** (`dryousuf_stage1_plus_nyu`: `261.1971 GiB` / `116,033` files)
- Constraints enforced: CPU-only (`1.0 vCPU / 2048 MiB`, 0 GPUs), zero bulk routing through local PC, zero source interference or deletions (`hasinishrak2015` / `hasinishrak74001`), zero global profile switching (`MODAL_PROFILE` explicit per process), ephemeral in-memory secret injection (`modal.Secret.from_dict`), strict NLBS cancer-label quarantine, and fresh-container SHA-256 + PNG decode verification.

---

## 2. Source Inventory, Deduplication & Quarantine Mapping

| Profile | Role | Active Apps (Start -> End) | Assets Supplied / Received | Existing -> Final Storage (`< 1 TiB` Free Tier) |
|---|---|---|---|---|
| `hasinishrak74001` | Source B (Preferred) | `0` -> `0` | Stage-1 RSNA `1520x912` (`54,709` files, `65.9501 GiB`), `splits_seed42.csv`, Mammo-CLIP `b5-model-best-epoch-7.tar` (`1.5405 GiB`), Stage-B checkpoints | Untouched (`0` deltas) |
| `hasinishrak2015` | Source A | `0` -> `0` (`ap-1WtcQA2...` stopped) | Stage-1 NLBS `1520x912` (`27,020` files, `35.9386 GiB`), `models/paper1/*` (`4` weights, `0.2573 GiB`), NYU `paper1_nyu_uint16_v1` (`34,294` files, `157.5057 GiB`) | Untouched (`0` compute/egress delta) |
| `tigerwood693` | Destination (DINOv2) | `0` -> `0` | **`81,731` files (`101.9755 GiB`)** | `122.62 -> ~224.60 GiB` (`USD 0.00/day`) |
| `tigerwood697` | Destination (ResNet-50) | `0` -> `0` | **`81,731` files (`101.9888 GiB`)** | `63.27 -> ~165.26 GiB` (`USD 0.00/day`) |
| `dryousufmozumder` | Destination (Option B) | `0` -> `0` | **`116,033` files (`261.1971 GiB`)** | `96.18 -> ~357.38 GiB` (`USD 0.00/day`) |

**Strict Quarantine & Deduplication Enforced in `src/utils/modal_transfer.py` (`assert_transfer_path_allowed`):**
- Hard-blocked anywhere in planning or worker execution: `exam_manifest.csv`, `datasets/nlbs/manifest.csv`, `nlbs/manifest.csv`, `/nlbs/raw/*` (`NLBSD_Metadata.csv`, `normal.7z`, `False Positive.7z`, `abnormal.7z`), `.modal.toml`, `git commit.md`, `memory/personal_info.md`.
- Excluded redundant `paper1_gmic_uint16_v1` (`159.37 GiB`, GMIC Steps 1–3 already locked locally) and heavy raw DICOM archives (`293.11 GiB` RSNA / `324.45 GiB` NLBS extracted) unless explicitly requested.

---

## 3. Pilot Concurrency Benchmark & TAR-Shard Architectural Evaluation

### 3.1 `1.5 GiB` Pilot (`tigerwood693`, Disjoint `~0.50 GiB` Slices + Resume Check)
| Pass | App ID | Config (`Containers × Streams = Workers`) | Files / GiB | Transfer Wall / Worker Speed | Verify Wall | Actual Billed Cost | Integrity |
|---|---|---|---:|---:|---:|---:|---|
| Pass 1 (`w16`) | `ap-E4yvjQNte1wT9JVwXXCBXu` | `8 × 2 = 16` | `745` / `0.4999 GiB` | `48.48s` (`10.56 / 13.02 MiB/s`) | `64.16s` | `USD 0.0226` | `745/745` PASS |
| Pass 2 (`w24`) | `ap-kXlrM2S1Qm2qYv8xf1sKev` | `12 × 2 = 24` | `821` / `0.4999 GiB` | `102.88s` (`4.98 / 5.79 MiB/s`) | `36.05s` | `USD 0.0325` | `821/821` PASS |
| **Pass 3 (`w32`)** | `ap-oKMx3fWX6a5k48fOPiEwGP` | **`16 × 2 = 32` (Pilot Winner)** | `826` / `0.4999 GiB` | **`34.69s` (`14.76 / 21.20 MiB/s`)** | **`30.52s`** | **`USD 0.0337`** | **`826/826` PASS** |
| Pass 4 (Resume) | `ap-DdNauV8XuFxWKMejih8DZ5` | `8 × 2 = 16` (Slice 0 re-run) | `0` new (`745` skipped) | `8.16s` (`1.07s` sum workers) | `24.78s` | `USD 0.0288` | `745/745` PASS |

- Settled Pilot delta across all 4 passes: **`USD 0.12`** credits drawn (`-3.15 -> -3.27`) vs `USD 0.15` cap; `0` file retries, `0` commit retries, **`USD 0.00000000`** delta on `hasinishrak2015` and `hasinishrak74001`.

### 3.2 `2–4 GiB` Uncompressed TAR Shards vs Direct `16 × 6` Async Streaming
- Evaluated source-side `2–4 GiB` TAR packaging + Modal Volumes `v2` (`--version=2`, Experimental) vs our existing `VOLUME_FS_VERSION_V1` direct async streaming.
- Observed empirically during the initial `10-min` `16 × 2` full run before user-requested pause: direct streaming already achieved **`~196 MiB/s`** (`117.78 GiB` in `10m`) on `dryousufmozumder` and **`~79–86 MiB/s`** (`46.53` / `50.85 GiB` in `10m`) on `tigerwood693`/`tigerwood697` with `USD 0.00` source cost.
- Rejected TAR packaging because it would consume scarce `hasinishrak2015` compute credits (`USD 7.50` remaining) and `+194 GiB` storage above `1 TiB`, require `+15–25 min` pack + `+15–20 min` unpack overhead, and require recreating `v2` volumes (`v1 -> v2` cannot convert in-place), whereas simply increasing intra-container async streams from `16 × 2` -> **`16 × 6 = 96` concurrent streams** (`adaptive_fallback -> 32` on retries) achieved **`100.22 MiB/s`** end-to-end resumed wall throughput on `tigerwood693` and **`~340 MiB/s` burst** on `dryousufmozumder`.

---

## 4. Implementation Details & Verification Engineering
- Added [`src/utils/modal_transfer.py`](../src/utils/modal_transfer.py): pure-Python `SourceSpec` / `TransferEntry` planner, strict NLBS/secret quarantine enforcement, LPT byte-balanced `partition_entries_by_bytes`, deterministic `select_pilot_entries` / `select_pilot_slice`, credential-safe `load_modal_profile_credentials` + `redact_secrets`, Modal rate cost projectors, and `async_transfer_shard` (`60s` + `SIGTERM`/`SIGINT` periodic `commit_volume_with_retry`, atomic `.part` -> `os.replace`, `peak_rss_mib`/`cpu_seconds` `rusage` telemetry, and automatic `96 -> 32` stream fallback on `file_retries >= 5` or `commit_retries >= 2`).
- Added [`scripts/transfer_modal_volumes_cloud.py`](../scripts/transfer_modal_volumes_cloud.py): `1.0 vCPU / 2048 MiB` CPU-only Modal app mounting destination `rsna-mammography-vol`, authenticating read-only source profiles inside containers via `modal.Client.from_credentials.aio` fed from ephemeral `modal.Secret.from_dict`, plus `--verify-only` mode running `ThreadPoolExecutor(max_workers=16)` inside each of the `16` containers (**`256` parallel SHA-256 + PNG decode verifiers**, speeding full verification from `35+ min` sequential FUSE reads down to **`176.86s` / `361.88s`**!).
- Added [`tests/test_transfer_modal_volumes_cloud.py`](../tests/test_transfer_modal_volumes_cloud.py): **`8/8` tests PASS** (`0.070–0.091s`), covering quarantine rejections, `74001`/`2015` source routing, byte-balanced sharding, disjoint pilot slicing, resume skip + redaction, cost math, and mocked `async_transfer_shard` streaming/commit verification.

---

## 5. Final Full-Transfer Verification & Settled Cost Accounting

| Destination | Bundle | Files Verified (`SHA-256` + `128` PNG Decodes) | Verify Time (`App ID`) | Pre-Transfer -> Final Credits Used | Total Transfer + Verify Cost | Status |
|---|---|---:|---|---:|---:|---|
| **`tigerwood693`** | `dinov2` (`101.9755 GiB`) | **`81,731 / 81,731`** (`0` err, `0` retries, `peak_rss=536 MiB`, `adaptive_fallback=False`) | `1,272.25s` (`ap-mXoD0F9Bi3OayKVEbgibSW`) | `-3.27 -> -3.80` (`-3.15` pre-pilot) | **`USD 0.53`** (`USD 0.65` w/ pilot) | **`all_verified: true`** |
| **`tigerwood697`** | `resnet50` (`101.9888 GiB`) | **`81,731 / 81,731`** (`0` err) | `176.86s` (`ap-yFsrplOAqFnUoOoqKQaPk0`) | `0.00 -> -0.58` | **`USD 0.58`** | **`all_verified: true`** |
| **`dryousufmozumder`** | `dryousuf_stage1_plus_nyu` (`261.1971 GiB`) | **`116,033 / 116,033`** (`0` err; `34,264` NYU + `81,694` Stage-1 + `9` models) | `361.88s` (`ap-Cn33ALCPpLzvqqCP2WhBPJ`) | `0.00 -> -0.71` | **`USD 0.71`** | **`all_verified: true`** |
| **`hasinishrak2015`** | Source A | Preserved untouched (`0` active) | — | `-22.50 -> -22.99` (`ephem 14.37798` / `egress 0.08029` unchanged; `+USD 0.49` UTC daily storage overage) | **`USD 0.00` compute/egress** | UNTOUCHED |
| **`hasinishrak74001`** | Source B | Preserved untouched (`0` active) | — | `-20.85 -> -20.85` (unchanged to 8 decimals) | **`USD 0.00000000`** | UNTOUCHED |
| **Combined Grand Total** | **3 Destinations** | **`279,495 / 279,495` verified** (`465.16 GiB`) | — | **All `< 1 TiB` (`USD 0.00/day` storage)** | **`USD 1.82`** (`USD 1.94` w/ pilot vs `7.16–10.77` cap) | **COMPLETE** |

Pinned weight hashes explicitly verified at destination:
- `models/paper1/dinov2_vits14_pretrain.pth`: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` (`88,283,115` B)
- `models/paper1/resnet50-11ad3fa6.pth`: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` (`102,540,417` B)
- `models/paper1/sample_image_model.p`: `9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014` (`24,705,061` B)
- `models/mammo_clip/b5-model-best-epoch-7.tar`: `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c` (`1,654,145,453` B)

*(Note: when running `scripts/run_paper1_nyu_modal.py --stage infer` on `dryousufmozumder`, create an empty `nlbs-mammography-vol` placeholder or guard the unused `/nlbs_vol` mount on `--stage infer` since Option B uses preprocessed `/rsna_vol/cache/paper1_nyu_uint16_v1/`.)*

---

## 6. Quality Assurance & Artifacts
- Execution & verification receipts committed under `reports/transfers/` (`pilot_*.json`, `full_tigerwood693_dinov2_*.json`, `full_tigerwood697_resnet50_*.json`, `full_dryousufmozumder_optb_*.json`).
- Offline suite: `python -m unittest discover -s tests -t . -p "test_transfer_modal_volumes_cloud.py" -v` -> **`8/8` PASS (`0.091s`)**.

---

## 7. Next Steps (Not Executed Yet)
- Await separate user approval before running `run_paper1_dinov2_modal.py` on `tigerwood693`, `run_paper1_resnet50_modal.py` on `tigerwood697`, or `run_paper1_nyu_modal.py` Step 2 RSNA->NLBS on `dryousufmozumder`.
