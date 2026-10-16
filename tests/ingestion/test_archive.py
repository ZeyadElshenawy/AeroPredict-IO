import zipfile
from pathlib import Path

import pytest

from flightops_ml.ingestion.archive import (
    EXPECTED_COLUMNS,
    ArchiveValidationError,
    validate_bts_archive,
)


def create_zip(
    path: Path,
    files: dict[str, list[str]],
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, columns in files.items():
            content = ",".join(columns) + "\n"
            archive.writestr(name, content)


def test_valid_archive_returns_csv_name(tmp_path: Path) -> None:
    archive_path = tmp_path / "valid.zip"
    create_zip(
        archive_path,
        {"T_ONTIME_REPORTING.csv": sorted(EXPECTED_COLUMNS)},
    )

    result = validate_bts_archive(archive_path)
    assert result == "T_ONTIME_REPORTING.csv"


def test_missing_archive_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_bts_archive(tmp_path / "missing.zip")


def test_corrupt_zip_raises_validation_error(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "corrupt.zip"
    archive_path.write_text("not a zip file", encoding="utf-8")

    with pytest.raises(ArchiveValidationError, match="Not a valid ZIP"):
        validate_bts_archive(archive_path)


def test_archive_with_missing_columns_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "missing_columns.zip"
    create_zip(archive_path, {"flights.csv": ["YEAR", "MONTH"]})

    with pytest.raises(ArchiveValidationError, match="Missing expected"):
        validate_bts_archive(archive_path)


def test_archive_with_multiple_csv_files_is_rejected(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "multiple.zip"
    create_zip(
        archive_path,
        {"first.csv": sorted(EXPECTED_COLUMNS), "second.csv": sorted(EXPECTED_COLUMNS)},
    )

    with pytest.raises(ArchiveValidationError, match="Expected exactly one CSV"):
        validate_bts_archive(archive_path)
