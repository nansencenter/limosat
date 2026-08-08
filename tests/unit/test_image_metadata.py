import importlib
import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


UTILS_PATH = Path(__file__).resolve().parents[2] / "limosat" / "utils.py"
SPEC = importlib.util.spec_from_file_location("limosat_utils_under_test", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)


def test_extract_date_supports_sentinel1_and_radarsat_names():
    assert UTILS.extract_date(
        "S1C_EW_GRDM_1SDH_20260723T101104_20260723T101209_008669.tiff"
    ) == pd.Timestamp("2026-07-23T10:11:04")
    assert UTILS.extract_date(
        "RS2_20260723_103309_0076_SCWA_HH_SGF_1303193.tiff"
    ) == pd.Timestamp("2026-07-23T10:33:09")


def test_extract_date_returns_none_for_unsupported_name():
    assert UTILS.extract_date("undated-image.tiff") is None


def test_image_processor_rejects_undated_images(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    image_processor = importlib.import_module("limosat.image_processor")
    from tests.factories import MatcherStub

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
