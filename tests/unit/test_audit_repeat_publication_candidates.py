import pandas as pd

from experiments import audit_repeat_publication_candidates as audit


def test_catalog_duplicate_requires_both_official_products(monkeypatch):
    primary = "S1B_EW_GRDM_1SDH_20200101T000000_20200101T000100_000001_AAAAAA_1111"
    repeat = "S1B_EW_GRDM_1SDH_20200101T000000_20200101T000100_000001_AAAAAA_2222"
    monkeypatch.setattr(audit, "official_grd_names", lambda key, timeout: [primary])
    candidates = pd.DataFrame(
        {
            "repeat_control_id": [audit.logical_product_key(primary)],
            "primary_product_name": [primary],
            "repeat_product_name": [repeat],
            "repeat_asf_url": ["https://example.invalid/repeat.zip"],
        }
    )

    result = audit.audit_candidates(candidates, timeout_seconds=1.0).iloc[0]

    assert result.primary_is_official_asf_grd_md
    assert not result.candidate_repeat_is_official_asf_grd_md
    assert result.candidate_audit_status == (
        "stale_catalog_duplicate_not_official_repeat"
    )
