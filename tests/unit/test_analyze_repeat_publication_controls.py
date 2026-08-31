import zipfile

from osgeo import gdal

from experiments.analyze_repeat_publication_controls import (
    compare_zip_archives,
    gcp_max_difference,
)


def write_test_zip(path, root, measurement=b"same pixels"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/manifest.safe", b"same manifest")
        archive.writestr(f"{root}/measurement/data.tiff", measurement)


def test_zip_comparison_ignores_repeat_publication_safe_root(tmp_path):
    primary = tmp_path / "primary.zip"
    repeat = tmp_path / "repeat.zip"
    write_test_zip(primary, "PRIMARY.SAFE")
    write_test_zip(repeat, "REPEAT.SAFE")

    result = compare_zip_archives(primary, repeat)

    assert result["normalized_member_names_equal"]
    assert result["normalized_members_with_size_or_crc_difference"] == 0
    assert result["measurement_members_with_size_or_crc_difference"] == 0
    assert not result["raw_zip_sha256_equal"]


def test_zip_comparison_detects_measurement_change(tmp_path):
    primary = tmp_path / "primary.zip"
    repeat = tmp_path / "repeat.zip"
    write_test_zip(primary, "PRIMARY.SAFE")
    write_test_zip(repeat, "REPEAT.SAFE", measurement=b"changed pixels")

    result = compare_zip_archives(primary, repeat)

    assert result["normalized_members_with_size_or_crc_difference"] == 1
    assert result["measurement_members_with_size_or_crc_difference"] == 1


def test_gcp_difference_includes_map_coordinates():
    left = [gdal.GCP(100.0, 200.0, 0.0, 10.0, 20.0)]
    right = [gdal.GCP(103.5, 200.0, 0.0, 10.0, 20.0)]

    assert gcp_max_difference(left, right) == 3.5
