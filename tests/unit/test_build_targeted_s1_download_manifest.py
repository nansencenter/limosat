from pathlib import Path

import pandas as pd

from experiments.build_targeted_s1_download_manifest import resolve_downloads


def test_resolved_downloads_are_sharded_under_requested_data_root():
    acquisitions = pd.DataFrame(
        {
            "priority_tier": [1, 2],
            "download_decision": [
                "ready_for_restore_or_download",
                "ready_for_restore_or_download",
            ],
            "standard_vae_pixels_local": [False, False],
            "sentinel1_product_name": ["S1_TEST_A", "S1_TEST_B"],
            "image_time": ["2020-01-02T03:00:00Z", "2020-04-02T03:00:00Z"],
            "image_id": [1, 2],
        }
    )
    out = resolve_downloads(
        acquisitions,
        {"S1_TEST_A": "https://example.test/S1_TEST_A.zip"},
        {},
        maximum_priority_tier=1,
        data_root=Path("/Volumes/KINGSTON/experiment"),
    )
    assert len(out) == 1
    assert out.iloc[0]["raw_zip_path"] == (
        "/Volumes/KINGSTON/experiment/sentinel1/raw/2020/01/S1_TEST_A.zip"
    )
