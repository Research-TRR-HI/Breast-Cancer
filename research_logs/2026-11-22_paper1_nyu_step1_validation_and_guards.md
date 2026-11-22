# Research Log — 2026-11-22: `[Paper 1]` NYU Step 1 Pre-Build Validation (`2vCPU/8GiB`), SHA-256 Resume & Storage Audit

- **Scope & Strict Gate:** Executed CPU-only config validation, SHA-256 resume hardening, CLI build guard verification, and read-only Modal volume storage audit (`ap-p0FrhTarpdGuiJaUUYKJ6Q`, ZERO GPUs allocated, ZERO GMIC/Stage-1 deletions, ZERO full build started).
- **Code & Test Hardening (`src/paper1/nyu_fourview_adapter.py`, `scripts/preprocess_paper1_nyu_cpu.py`, `tests/test_paper1_nyu_step1_cpu.py`):**
  1. Added `verify_and_load_nyu_resumed_records`: enforces `png_sha256` + `u16_sha256` + `file_bytes` verification on resume (rejecting same-size byte corruption and missing checksums, unit-tested in `test_4`).
  2. Added `validate_nyu_cli_stage_and_guard`: implements guarded `--stage build`, `--stage verify`, `--stage validate_config`, blocking unapproved `--stage build` cleanly unless `--approve-full-build` is explicitly passed (`test_5` + live Modal CLI exit `1` check verified).
  3. Sized production worker `process_nyu_cpu_shard` to **`cpu=2.0, memory=8192`** and added `verify_nyu_cpu_cache`.
- **Live Cloud Validation (`ap-p0FrhTarpdGuiJaUUYKJ6Q`, `160` representative imgs = `40` exams):**
  - `88.874s` (`1.800 img/s`), peak RSS = **`5.117 GB`** (`2.883 GiB` headroom below `8 GiB`), measured cost = **`$0.00392`**.
  - Live SHA-256 resume tamper test passed (`158` resumed, `2` tampered/deleted rejected and repaired cleanly to `160/160`).
  - Read-only Volume usage: `rsna-mammography-vol = 559.204 GiB` (`159.371 GiB` GMIC), `nlbs-mammography-vol = 409.010 GiB` -> **`968.214 GiB`** combined Paper 1 + `16` older LLM/thesis volumes on `hasinishrak2015` (exceeding Modal Starter's **1 TiB / 1,024 GiB** free tier by `~$1.02` so far).
  - Weighted NYU Step 1 projections (`34,264` imgs, `12` shards/split + `25s` container overhead): **`~$0.869 USD`** compute (`$0.2464` RSNA + `$0.6186` NLBS + `~$0.004` verify) and **`137.00 GiB`** storage (`19.48 GiB` RSNA + `117.52 GiB` NLBS = **`~$0.41/day`** / `~$3.29` rest of Nov / `$12.33/mo` beyond 1 TiB).
