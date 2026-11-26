# MIAS v1.21 Official Download & Raw Integrity Audit

**Dataset:** Mammographic Image Analysis Society (MIAS) database v1.21 (25 August 2015)  
**Repository:** University of Cambridge Apollo (`https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459`)  
**DOI:** `https://doi.org/10.17863/CAM.105113` (Handle: `1810/250394`)  
**Licence:** Attribution 2.0 UK: England & Wales (`http://creativecommons.org/licenses/by/2.0/uk/`)  
**Modal Profile / Volume:** `tigerwood697` / `rsna-mammography-vol` (`/external_datasets/mias/raw/`)  
**Modal Execution:** CPU-only (`cpu=1.0, memory=2048`, **0 GPUs**), App `paper1-exp3-mias-cpu` (`ap-mSv34xiz2PKgwnelAXWite`)  
**Script:** `scripts/download_and_audit_mias_modal.py`

---

## 1. DSpace 7 REST API Bitstream Inventory & Transfer Metrics

Queried `https://api.repository.cam.ac.uk/server/api/core/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459/bundles`:

| Bundle | Filename | Size (Bytes) | Expected MD5 | Verified MD5 | SHA-256 | Elapsed (s) | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ORIGINAL` | `miasdbv1.21.zip` | `1,617,327,652` (~1.506 GiB) | `d790185baed18ab199efc85388b75dfc` | `d790185baed18ab199efc85388b75dfc` (MATCH) | `2bf4f7d68d1fa4fae2f9e0edc4da2b1a1aa18986d4a30be5d7f53572cc2713f4` | `29.817` | `51.729 MB/s` |
| `LICENSE` | `license.txt` | `3,893` | `87eda9de84448d1f82354d60eee3eb5f` | `87eda9de84448d1f82354d60eee3eb5f` (MATCH) | `db610c9837ab13186d8143376ebe891b0c2fccfa7a2dbd6ad67275582ad27350` | `0.122` | `0.030 MB/s` |
| `CC-LICENSE` | `license_rdf` | `1,370` | `e9fbe0c18c54a5a24f2661023416007e` | `e9fbe0c18c54a5a24f2661023416007e` (MATCH) | `a3ac876433dea1b3031c7779ecae1105eb5af1c2d4fc6920be85609dc5534a8e` | `0.120` | `0.011 MB/s` |

- **Total Downloaded:** `1,617,332,915` bytes (`1.506 GiB`) in `~30.06s` transfer time.
- **Estimated Modal Compute Cost:** `< $0.005` (`1 vCPU, 2 GiB RAM`, ~2 minutes total container runtime including image build, extraction, and multiprocessing PGM audit; **0 GPU seconds**).

---

## 2. Extracted Archive & Multiprocessing PGM Verification

Extracted to `/external_datasets/mias/raw/extracted/MIASDBv1.21/`:
- **Total files extracted:** `324` (`322` `.pgm` images + `2` PDFs: `00README.pdf` [`120,081` bytes, SHA-256 `02d9334b20bfe0de0d1e393e99c02c52395935dc6ef913f62a2a0c1839aad595`] and `MIASDB Excerpta Medica 1994.pdf`).
- **PGM Decode Status:** `322 / 322` decoded cleanly (`decode_ok = True`), `0` errors.
- **PGM Header Format:** `100%` binary `P5`, `maxval = 255`, `dtype = uint8` (`8-bit`), `0` comment lines.
- **Pixel Value Range:** `pixel_min` in `[0, 3]`, `pixel_max` in `[224, 255]`.
- **Duplicate Check:** `0` duplicate file SHA-256 hashes; `0` duplicate pixel-payload SHA-256 hashes (`322` unique images).
- **Dimensions & Filename Suffix Mapping (50-micron digitized film):**
  - `s` suffix: `4320 × 1600` (`48` images)
  - `m` suffix: `4320 × 2048` (`101` images)
  - `l` suffix: `4320 × 2600` (`149` images)
  - `x` suffix: `5200 × 4000` (`24` images)
- **Orientation Finding (CRITICAL):** Raw MIAS v1.21 PGM rasters are **landscape (`width > height`)** with the chest wall oriented toward the **top** edge (`top_half_mean >> bottom_half_mean`, e.g., `mdb001lm`: `111.77` vs `32.76`; `mdb003ll`: `139.93` vs `27.04`). They are rotated 90° relative to standard portrait FFDM mammograms (`1520 × 912` / `2944 × 1920`).
- **Raw Audit JSON Artifact:** `reports/paper1/expanded/exp3/imdb_mias/mias_raw_audit.json` (and mirrored at `/cache/mias/mias_raw_audit.json` on `rsna-mammography-vol`).
