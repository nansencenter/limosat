import json

import pytest
import pystac


@pytest.mark.unit
def test_build_stac_item_collection_empty_files():
    from limosat.catalog import build_stac_item_collection

    with pytest.raises(ValueError, match="No files provided"):
        build_stac_item_collection([])


@pytest.mark.unit
def test_build_stac_item_collection_missing_file():
    from limosat.catalog import build_stac_item_collection

    with pytest.raises(FileNotFoundError, match="Missing file: /nonexistent/file.tiff"):
        build_stac_item_collection(["/nonexistent/file.tiff"], check_exists=True)


@pytest.mark.unit
def test_build_stac_item_collection_writes_sorted_catalog(tmp_path):
    from limosat.catalog import build_stac_item_collection

    file_early = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_AAA1.tiff"
    file_late = tmp_path / "S1B_EW_GRDM_1SDH_20250101T000100_20250101T000130_012346_ABCDEF_BBB2.tiff"
    file_early.touch()
    file_late.touch()

    out_path = tmp_path / "catalog.json"
    coll = build_stac_item_collection([str(file_late), str(file_early)], out_path=str(out_path))

    assert isinstance(coll, pystac.ItemCollection)
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["type"] == "FeatureCollection"
    ids = [feature["id"] for feature in payload["features"]]
    assert ids == [file_early.stem, file_late.stem]

    first_props = payload["features"][0]["properties"]
    assert first_props["image_id"] == 1
    assert first_props["scene_id"] == file_early.stem
    assert first_props["product_uid"] == "AAA1"
    assert first_props["filename"] == file_early.name
    assert first_props["filepath"] == str(file_early)
    assert payload["features"][0]["assets"]["image"]["href"] == str(file_early)


@pytest.mark.unit
def test_build_stac_item_collection_keeps_uid_collisions_with_different_scene_stems(tmp_path):
    from limosat.catalog import build_stac_item_collection

    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_DUP1.tiff"
    file2 = tmp_path / "S1B_EW_GRDM_1SDH_20250102T000000_20250102T000030_012346_ABCDEF_DUP1.tiff"
    file1.touch()
    file2.touch()

    out_path = tmp_path / "catalog.json"
    coll = build_stac_item_collection([str(file1), str(file2)], out_path=str(out_path))
    features = coll.to_dict()["features"]

    assert len(features) == 2
    assert [feature["id"] for feature in features] == [file1.stem, file2.stem]
    assert [feature["properties"]["product_uid"] for feature in features] == ["DUP1", "DUP1"]


@pytest.mark.unit
def test_build_stac_item_collection_rejects_duplicate_scene_stem(tmp_path):
    from limosat.catalog import build_stac_item_collection

    scene_name = "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_DUP1.tiff"
    dir1 = tmp_path / "a"
    dir2 = tmp_path / "b"
    dir1.mkdir()
    dir2.mkdir()
    file1 = dir1 / scene_name
    file2 = dir2 / scene_name
    file1.touch()
    file2.touch()

    with pytest.raises(ValueError, match="Duplicate Sentinel-1 scene id"):
        build_stac_item_collection([str(file1), str(file2)])


@pytest.mark.unit
def test_build_stac_item_collection_orders_by_datetime_then_name(tmp_path):
    from limosat.catalog import build_stac_item_collection

    later = tmp_path / "S1A_EW_GRDM_1SDH_20250102T000000_20250102T000030_012350_ABCDEF_LATE.tiff"
    same_dt_a = tmp_path / "S1A_EW_GRDM_1SDH_20250101T010000_20250101T010030_012347_ABCDEF_AAAA.tiff"
    same_dt_b = tmp_path / "S1B_EW_GRDM_1SDH_20250101T010000_20250101T010030_012348_ABCDEF_BBBB.tiff"
    later.touch()
    same_dt_a.touch()
    same_dt_b.touch()

    out_path = tmp_path / "catalog.json"
    build_stac_item_collection(
        [str(later), str(same_dt_b), str(same_dt_a)],
        out_path=str(out_path),
    )

    payload = json.loads(out_path.read_text())
    ids = [feature["id"] for feature in payload["features"]]
    image_ids = [feature["properties"]["image_id"] for feature in payload["features"]]

    assert ids == [same_dt_a.stem, same_dt_b.stem, later.stem]
    assert image_ids == [1, 2, 3]


@pytest.mark.unit
def test_build_stac_item_collection_cleans_temp_file(tmp_path):
    from limosat.catalog import build_stac_item_collection

    file1 = tmp_path / "S1A_EW_GRDM_1SDH_20250101T000000_20250101T000030_012345_ABCDEF_KEEP.tiff"
    file1.touch()

    out_path = tmp_path / "catalog.json"
    build_stac_item_collection([str(file1)], out_path=str(out_path))

    assert not (tmp_path / "catalog.json.tmp").exists()
    assert out_path.exists()
