import csv
import io
import zipfile
from pathlib import Path

EXPECTED_COLUMNS = {
    "YEAR",
    "MONTH",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DEL15",
    "CANCELLED",
    "DIVERTED",
}


class ArchiveValidationError(ValueError):
    """Raised when a source archive fails validation."""


def validate_bts_archive(archive_path: Path) -> str:
    """Validate a BTS ZIP and return its single CSV member name."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ArchiveValidationError(f"Corrupt ZIP member: {bad_member}")

            csv_files = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]

            if len(csv_files) != 1:
                raise ArchiveValidationError(
                    f"Expected exactly one CSV, found {len(csv_files)}"
                )
            csv_name = csv_files[0]

            with archive.open(csv_name) as binary_file:
                text_file = io.TextIOWrapper(
                    binary_file,
                    encoding="utf-8-sig",
                    newline="",
                )
                reader = csv.reader(text_file)
                header = next(reader, None)

            if header is None:
                raise ArchiveValidationError("CSV is empty")

            missing = EXPECTED_COLUMNS - set(header)
            if missing:
                raise ArchiveValidationError(
                    f"Missing expected columns: {sorted(missing)}"
                )

            return csv_name

    except zipfile.BadZipFile as exc:
        raise ArchiveValidationError(
            f"Not a valid ZIP archive: {archive_path}"
        ) from exc
