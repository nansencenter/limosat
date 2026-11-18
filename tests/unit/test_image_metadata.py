import pandas as pd
import pytest

import limosat.image_processor as image_processor
from limosat.utils import extract_date
from tests.factories import MatcherStub


def test_extract_date_supports_sentinel1_and_radarsat_names():
    assert extract_date(
        "S1C_EW_GRDM_1SDH_20260723T101104_20260723T101209_008669.tiff"
    ) == pd.Timestamp("2026-07-23T10:11:04")
    assert extract_date(
        "RS2_20260723_103309_0076_SCWA_HH_SGF_1303193.tiff"
    ) == pd.Timestamp("2026-07-23T10:33:09")


def test_extract_date_returns_none_for_unsupported_name():
    assert extract_date("undated-image.tiff") is None


def test_image_processor_rejects_undated_images(monkeypatch):
    class UndatedImage:
        date = None

    monkeypatch.setattr(image_processor, "Image", lambda filename: UndatedImage())
    processor = image_processor.ImageProcessor(
        points=image_processor.Keypoints(),
        model=None,
        matcher=MatcherStub(),
        persist_updates=False,
    )

    with pytest.raises(ValueError, match="Could not determine image acquisition time"):
        processor.process_image(1, "undated-image.tiff")
