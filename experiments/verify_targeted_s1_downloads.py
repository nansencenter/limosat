#!/usr/bin/env python3
"""Verify targeted Sentinel-1 archives against the download URL manifest."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL_FILE = (
    ROOT / "results/iabp_s1_stratified_coverage/tier1_asf_urls.txt"
)
DEFAULT_DOWNLOAD_ROOT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/"
    "limosat_descriptor_update_2020/sentinel1/raw"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/iabp_s1_stratified_coverage"


def expected_names(url_file: Path) -> list[str]:
    names = [
        Path(urlparse(line.strip()).path).name
        for line in url_file.read_text().splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError("Download URL manifest contains duplicate archive names")
    return names


def verify_archive(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        return {
            "zip_ok": False,
            "zip_error": str(error),
            "member_count": None,
            "uncompressed_bytes": None,
        }
    return {
        "zip_ok": corrupt_member is None,
        "zip_error": None if corrupt_member is None else f"CRC failure: {corrupt_member}",
        "member_count": len(members),
        "uncompressed_bytes": sum(member.file_size for member in members),
    }


def audit_downloads(
    url_file: Path, download_root: Path, verify_zip_contents: bool = True
) -> tuple[dict[str, object], list[dict[str, object]]]:
    expected = expected_names(url_file)
    expected_set = set(expected)
    archives = [
        path
        for path in download_root.rglob("*.zip")
        if not path.name.startswith("._")
    ]
    archive_by_name: dict[str, list[Path]] = {}
    for path in archives:
        archive_by_name.setdefault(path.name, []).append(path)

    rows: list[dict[str, object]] = []
    for name in expected:
        paths = archive_by_name.get(name, [])
        row: dict[str, object] = {
            "archive_name": name,
            "present": len(paths) == 1,
            "path": str(paths[0]) if len(paths) == 1 else None,
            "compressed_bytes": paths[0].stat().st_size if len(paths) == 1 else None,
            "zip_ok": None,
            "zip_error": None,
            "member_count": None,
            "uncompressed_bytes": None,
        }
        if len(paths) == 1 and verify_zip_contents:
            row.update(verify_archive(paths[0]))
        rows.append(row)

    missing = sorted(name for name in expected if name not in archive_by_name)
    duplicates = sorted(name for name, paths in archive_by_name.items() if len(paths) > 1)
    unexpected = sorted(name for name in archive_by_name if name not in expected_set)
    partials = sorted(
        str(path)
        for path in download_root.rglob("*.part")
        if not path.name.startswith("._")
    )
    partial_metadata_sidecars = sorted(
        str(path) for path in download_root.rglob("._*.part")
    )
    metadata_sidecars = sorted(
        str(path) for path in download_root.rglob("._*.zip")
    )
    bad_archives = sorted(
        str(row["archive_name"])
        for row in rows
        if row["zip_ok"] is False
    )
    downloaded_bytes = sum(
        int(row["compressed_bytes"])
        for row in rows
        if row["compressed_bytes"] is not None
    )
    complete = not (missing or duplicates or unexpected or partials or bad_archives)
    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "url_manifest": str(url_file),
        "download_root": str(download_root),
        "download_root_is_kingston": str(download_root).startswith("/Volumes/KINGSTON/"),
        "zip_contents_verified": verify_zip_contents,
        "expected_archives": len(expected),
        "present_expected_archives": sum(bool(row["present"]) for row in rows),
        "downloaded_bytes": downloaded_bytes,
        "missing_archives": missing,
        "duplicate_archives": duplicates,
        "unexpected_archives": unexpected,
        "partial_files": partials,
        "bad_archives": bad_archives,
        "macos_metadata_sidecars": len(metadata_sidecars),
        "macos_partial_metadata_sidecars": len(partial_metadata_sidecars),
        "complete": complete,
    }
    return summary, rows


def write_outputs(
    summary: dict[str, object],
    rows: list[dict[str, object]],
    output_dir: Path,
    output_prefix: str = "tier1",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_prefix}_download_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output_dir / f"{output_prefix}_download_verification.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="tier1")
    parser.add_argument(
        "--skip-zip-contents",
        action="store_true",
        help="Check names and sizes without decompressing every member for CRC validation.",
    )
    args = parser.parse_args()
    summary, rows = audit_downloads(
        args.url_file,
        args.download_root,
        verify_zip_contents=not args.skip_zip_contents,
    )
    write_outputs(summary, rows, args.output_dir, args.output_prefix)
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
