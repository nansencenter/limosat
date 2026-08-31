import csv
import zipfile

from experiments.verify_targeted_s1_downloads import audit_downloads, write_outputs


def test_audit_verifies_expected_zip_and_ignores_macos_sidecar(tmp_path):
    download_root = tmp_path / "sentinel1" / "raw" / "2020" / "01"
    download_root.mkdir(parents=True)
    archive_path = download_root / "S1_TEST.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("S1_TEST.SAFE/manifest.safe", "test")
    (download_root / "._S1_TEST.zip").write_bytes(b"metadata")
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.test/GRD_MD/SA/S1_TEST.zip\n")

    summary, rows = audit_downloads(url_file, tmp_path / "sentinel1" / "raw")

    assert summary["complete"] is True
    assert summary["expected_archives"] == 1
    assert summary["present_expected_archives"] == 1
    assert summary["macos_metadata_sidecars"] == 1
    assert summary["bad_archives"] == []
    assert rows[0]["zip_ok"] is True


def test_audit_fails_when_expected_download_is_only_partial(tmp_path):
    download_root = tmp_path / "raw"
    download_root.mkdir()
    (download_root / "S1_MISSING.zip.part").write_bytes(b"partial")
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.test/S1_MISSING.zip\n")

    summary, _ = audit_downloads(url_file, download_root)

    assert summary["complete"] is False
    assert summary["missing_archives"] == ["S1_MISSING.zip"]
    assert summary["partial_files"] == [
        str(download_root / "S1_MISSING.zip.part")
    ]


def test_audit_ignores_macos_partial_sidecar(tmp_path):
    download_root = tmp_path / "raw"
    download_root.mkdir()
    archive_path = download_root / "S1_TEST.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("S1_TEST.SAFE/manifest.safe", "test")
    (download_root / "._S1_TEST.zip.part").write_bytes(b"metadata")
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.test/S1_TEST.zip\n")

    summary, _ = audit_downloads(url_file, download_root)

    assert summary["complete"] is True
    assert summary["partial_files"] == []
    assert summary["macos_partial_metadata_sidecars"] == 1


def test_write_outputs_uses_requested_prefix(tmp_path):
    summary = {"complete": True}
    rows = [{"archive_name": "S1_TEST.zip", "present": True}]

    write_outputs(summary, rows, tmp_path, output_prefix="full70")

    assert (tmp_path / "full70_download_verification.json").is_file()
    with (tmp_path / "full70_download_verification.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"archive_name": "S1_TEST.zip", "present": "True"}
        ]
