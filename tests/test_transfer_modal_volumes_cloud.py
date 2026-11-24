"""Offline CPU Unit Tests for Cloud-to-Cloud Modal Volume Transfer Infrastructure."""

from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.modal_transfer import (
    TransferEntry,
    assert_transfer_path_allowed,
    estimate_storage_cost_usd,
    estimate_transfer_compute_cost_usd,
    estimate_worker_hourly_cost_usd,
    get_bundle_specs,
    load_modal_profile_credentials,
    map_source_file_to_dest,
    partition_entries_by_bytes,
    project_full_transfer_with_overhead,
    redact_secrets,
    select_pilot_entries,
    select_pilot_slice,
    should_skip_existing_file,
)


class TestModalTransferQuarantineAndPlanning(unittest.TestCase):
    def test_quarantine_blocks_all_label_and_secret_paths(self) -> None:
        blocked = [
            "datasets/nlbs/exam_manifest.csv",
            "/datasets/nlbs/manifest.csv",
            "nlbs/manifest.csv",
            "nlbs/raw/NLBSD_Metadata.csv",
            "/nlbs/raw/normal.7z",
            "nlbs/raw/False Positive.7z",
            "nlbs/raw/abnormal.7z",
            "~/.modal.toml",
            "memory/personal_info.md",
            "git commit.md",
        ]
        for path in blocked:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    assert_transfer_path_allowed(path, context="unit_test")

    def test_allowed_label_free_paths_pass(self) -> None:
        allowed = [
            "/cache/stage1_rsna_uint16/png_1520x912/10006_1459541791.png",
            "/cache/stage1_nlbs_uint16/png_1520x912/1.2.840.png",
            "models/paper1/dinov2_vits14_pretrain.pth",
            "models/paper1/resnet50-11ad3fa6.pth",
            "nlbs/extracted/study_1/img.dcm",
        ]
        for path in allowed:
            self.assertTrue(assert_transfer_path_allowed(path, context="unit_test"))

    def test_bundles_prefer_74001_for_rsna_stage1(self) -> None:
        glam_specs = get_bundle_specs("glam_nlbs")
        self.assertEqual(len(glam_specs), 1)
        self.assertEqual(glam_specs[0].src_profile, "hasinishrak2015")
        self.assertEqual(glam_specs[0].src_prefix, "cache/paper1_gmic_uint16_v1/nlbs")
        self.assertEqual(glam_specs[0].dst_prefix, "cache/paper1_gmic_uint16_v1/nlbs")

        for bundle_name, expected_weight in [
            ("dinov2", "models/paper1/dinov2_vits14_pretrain.pth"),
            ("resnet50", "models/paper1/resnet50-11ad3fa6.pth"),
        ]:
            specs = get_bundle_specs(bundle_name)
            by_prefix = {s.src_prefix: s for s in specs}
            self.assertEqual(by_prefix["cache/stage1_rsna_uint16"].src_profile, "hasinishrak74001")
            self.assertEqual(by_prefix["rsna/splits_seed42.csv"].src_profile, "hasinishrak74001")
            self.assertEqual(by_prefix["cache/stage1_nlbs_uint16"].src_profile, "hasinishrak2015")
            self.assertIn(expected_weight, by_prefix)
            self.assertEqual(by_prefix[expected_weight].src_profile, "hasinishrak2015")

    def test_map_source_file_to_dest_handles_exact_and_subtree(self) -> None:
        specs = get_bundle_specs("dinov2")
        rsna_spec = next(s for s in specs if s.src_prefix == "cache/stage1_rsna_uint16")
        w_spec = next(s for s in specs if s.src_prefix == "models/paper1/dinov2_vits14_pretrain.pth")
        self.assertEqual(
            map_source_file_to_dest("/cache/stage1_rsna_uint16/png_1520x912/a.png", rsna_spec),
            "cache/stage1_rsna_uint16/png_1520x912/a.png",
        )
        self.assertEqual(
            map_source_file_to_dest("/models/paper1/dinov2_vits14_pretrain.pth", w_spec),
            "models/paper1/dinov2_vits14_pretrain.pth",
        )

    def test_partition_entries_by_bytes_preserves_all_items_balanced(self) -> None:
        entries = [
            TransferEntry("p1", "vol", f"dir/f{i}.png", "vol", f"dir/f{i}.png", 1000 + (i * 137) % 500)
            for i in range(64)
        ]
        shards = partition_entries_by_bytes(entries, num_shards=8)
        self.assertEqual(len(shards), 8)
        flattened = [item.dst_path for s in shards for item in s]
        self.assertEqual(len(flattened), 64)
        self.assertEqual(len(set(flattened)), 64)
        shard_bytes = [sum(i.size_bytes for i in s) for s in shards]
        self.assertLess(max(shard_bytes) - min(shard_bytes), 2000)

    def test_select_pilot_entries_covers_both_sources_within_budget(self) -> None:
        gib = 1024**3
        entries = [
            TransferEntry("hasinishrak74001", "v", f"cache/rsna/{i}.png", "v", f"cache/rsna/{i}.png", 2 * 1024 * 1024)
            for i in range(500)
        ] + [
            TransferEntry("hasinishrak2015", "v", f"cache/nlbs/{i}.png", "v", f"cache/nlbs/{i}.png", 2 * 1024 * 1024)
            for i in range(500)
        ] + [
            TransferEntry("hasinishrak2015", "v", "models/paper1/dinov2.pth", "v", "models/paper1/dinov2.pth", 88 * 1024 * 1024)
        ]
        pilot = select_pilot_entries(entries, max_gib=1.25)
        total = sum(e.size_bytes for e in pilot)
        self.assertLessEqual(total, int(1.25 * gib))
        self.assertGreater(total, int(1.10 * gib))
        sources = {e.src_profile for e in pilot}
        self.assertEqual(sources, {"hasinishrak74001", "hasinishrak2015"})

        s0 = select_pilot_slice(pilot, slice_index=0, num_slices=3)
        s1 = select_pilot_slice(pilot, slice_index=1, num_slices=3)
        s2 = select_pilot_slice(pilot, slice_index=2, num_slices=3)
        combined = [e.dst_path for s in (s0, s1, s2) for e in s]
        self.assertEqual(len(combined), len(pilot))
        self.assertEqual(len(set(combined)), len(pilot))

        proj = project_full_transfer_with_overhead(
            total_bytes=102 * (1024**3),
            measured_worker_throughput_mbps=150.0,
            num_containers=12,
        )
        self.assertGreater(proj["total_container_hours"], proj["xfer_container_hours"])
        self.assertGreater(proj["estimated_total_usd"], 0.0)

    def test_async_transfer_shard_streams_and_commits_cleanly(self) -> None:
        import asyncio
        import os
        from unittest.mock import AsyncMock
        from src.utils.modal_transfer import async_transfer_shard

        class _FakeReadFile:
            @staticmethod
            async def aio(path: str):
                yield b"hello-"
                yield path.encode("utf-8")

        class _FakeSrcVol:
            read_file = _FakeReadFile()

        class _FakeCommit:
            def __init__(self) -> None:
                self.calls = 0

            async def aio(self) -> None:
                self.calls += 1

        class _FakeDstVol:
            def __init__(self) -> None:
                self.commit = _FakeCommit()

        with tempfile.TemporaryDirectory() as tmpdir:
            mount = Path(tmpdir)
            payload = b"hello-cache/stage1_rsna_uint16/a.png"
            existing = mount / "cache/stage1_rsna_uint16/skip.png"
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"already-here")
            items = [
                {
                    "src_profile": "hasinishrak74001",
                    "src_volume": "rsna-mammography-vol",
                    "src_path": "cache/stage1_rsna_uint16/a.png",
                    "dst_path": "cache/stage1_rsna_uint16/a.png",
                    "size_bytes": len(payload),
                },
                {
                    "src_profile": "hasinishrak74001",
                    "src_volume": "rsna-mammography-vol",
                    "src_path": "cache/stage1_rsna_uint16/skip.png",
                    "dst_path": "cache/stage1_rsna_uint16/skip.png",
                    "size_bytes": len(b"already-here"),
                },
            ]
            dst_vol = _FakeDstVol()
            fake_env = {
                "SRC_TOKEN_ID_HASINISHRAK74001": "ak-dummy",
                "SRC_TOKEN_SECRET_HASINISHRAK74001": "as-dummy",
            }
            from unittest.mock import patch

            with patch.dict(os.environ, fake_env, clear=False):
                res = asyncio.run(
                    async_transfer_shard(
                        job_id="unit_job",
                        shard_id=0,
                        entries=items,
                        dst_vol=dst_vol,
                        mount_root=mount,
                        streams_per_container=6,
                        commit_interval_s=30.0,
                        client_factory=AsyncMock(return_value=object()),
                        volume_factory=lambda *args, **kwargs: _FakeSrcVol(),
                    )
                )
            self.assertEqual(res["stats"]["transferred_files"], 1)
            self.assertEqual(res["stats"]["skipped_files"], 1)
            self.assertEqual(res["stats"]["failed_files"], 0)
            self.assertEqual((mount / "cache/stage1_rsna_uint16/a.png").read_bytes(), payload)
            self.assertGreaterEqual(dst_vol.commit.calls, 1)

    def test_resume_skip_and_secret_redaction_and_cost_math(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "sample.png"
            f.write_bytes(b"abcd")
            self.assertTrue(should_skip_existing_file(f, 4, None))
            self.assertFalse(should_skip_existing_file(f, 5, None))
            self.assertTrue(should_skip_existing_file(f, 4, {"size_bytes": 4, "sha256": "abc", "verified": True}))
            self.assertFalse(should_skip_existing_file(f, 4, {"size_bytes": 4, "sha256": "", "verified": False}))

            toml = root / ".modal.toml"
            toml.write_text(
                '[p1]\ntoken_id = "ak-test1234567890"\ntoken_secret = "as-secret987654321"\n',
                encoding="utf-8",
            )
            creds = load_modal_profile_credentials(["p1"], modal_toml_path=toml)
            self.assertEqual(creds["p1"], ("ak-test1234567890", "as-secret987654321"))
            cleaned = redact_secrets("failed auth ak-test1234567890 as-secret987654321")
            self.assertNotIn("ak-test1234567890", cleaned)
            self.assertNotIn("as-secret987654321", cleaned)

        self.assertAlmostEqual(estimate_worker_hourly_cost_usd(1.0, 2048), 0.0633, places=5)
        compute = estimate_transfer_compute_cost_usd(1024**3, 120.0, 8)
        self.assertGreater(compute["estimated_cost_usd"], 0.0)
        under = estimate_storage_cost_usd(900.0)
        self.assertEqual(under["overage_gib"], 0.0)
        self.assertEqual(under["daily_usd"], 0.0)
        over = estimate_storage_cost_usd(1124.0)
        self.assertAlmostEqual(over["overage_gib"], 100.0, places=4)
        self.assertAlmostEqual(over["monthly_usd"], 9.0, places=4)


if __name__ == "__main__":
    unittest.main()
