import csv
import zipfile
from pathlib import Path

import pytest
from scripts import ingest_bts_year as orchestrator


def create_month_archive(
    path: Path,
    year: int = 2025,
    month: int = 1,
    *,
    include_all_days: bool = True,
) -> Path:
    """Create a small synthetic BTS archive for validation tests."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(orchestrator.SOURCE_COLUMNS)
    rows = []

    from calendar import monthrange

    day_count = monthrange(year, month)[1] if include_all_days else 1

    for day in range(1, day_count + 1):
        row = {column: "1" for column in fieldnames}
        row["YEAR"] = str(year)
        row["MONTH"] = str(month)
        row["FL_DATE"] = f"{month:02d}/{day:02d}/{year} 12:00:00 AM"
        rows.append(row)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        with archive.open("T_ONTIME_REPORTING.csv", "w") as binary_file:
            import io

            with io.TextIOWrapper(
                binary_file,
                encoding="utf-8",
                newline="",
            ) as text_file:
                writer = csv.DictWriter(text_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    return path


def test_validate_month_contents_accepts_complete_month(tmp_path):
    archive = create_month_archive(tmp_path / "2025-01.zip")

    rows = orchestrator.validate_month_contents(archive, 2025, 1)

    assert rows == 31


def test_validate_month_contents_rejects_incomplete_month(tmp_path):
    archive = create_month_archive(
        tmp_path / "2025-01.zip",
        include_all_days=False,
    )

    with pytest.raises(ValueError, match="distinct calendar dates"):
        orchestrator.validate_month_contents(archive, 2025, 1)


def test_process_month_skips_verified_existing_load(tmp_path, monkeypatch):
    create_month_archive(tmp_path / "2025" / "2025-01.zip")

    monkeypatch.setattr(
        orchestrator,
        "get_database_state",
        lambda checksum, year, month: (31, 31, True),
    )
    monkeypatch.setattr(
        orchestrator,
        "load_month",
        lambda *args: pytest.fail("Existing month must not be loaded"),
    )

    import requests

    with requests.Session() as session:
        result = orchestrator.process_month(
            session,
            tmp_path,
            2025,
            1,
            execute=True,
        )

    assert result == "skipped"


def test_process_month_dry_run_does_not_download_missing_archive(
    tmp_path,
    monkeypatch,
):
    import requests

    monkeypatch.setattr(
        orchestrator,
        "download_month",
        lambda *args: pytest.fail("Dry run must not download"),
    )

    with requests.Session() as session:
        result = orchestrator.process_month(
            session,
            tmp_path,
            2025,
            3,
            execute=False,
        )

    assert result == "planned"
    assert not (tmp_path / "2025" / "2025-03.zip").exists()


class FakeCursor:
    """Provide predetermined database query results."""

    def __init__(self, results):
        self.results = iter(results)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        pass

    def fetchone(self):
        return next(self.results)


class FakeConnection:
    """Provide a mocked database connection."""

    def __init__(self, results):
        self.cursor_instance = FakeCursor(results)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.cursor_instance


def test_database_state_rejects_incomplete_row_number_range(monkeypatch):
    # 31 rows exist, but the first/last row numbers reveal a gap.
    fake_connection = FakeConnection(
        [
            (31, 1, 32),  # staged count, minimum row number, maximum row number
            (31, 31, 0),  # records_seen, records_inserted, records_skipped
        ]
    )

    monkeypatch.setattr(
        orchestrator,
        "database_connection",
        lambda: fake_connection,
    )

    staged_rows, audited_rows, matches = orchestrator.get_database_state(
        "fake-checksum",
        2025,
        1,
    )

    assert staged_rows == 31
    assert audited_rows == 31
    assert matches is False


def test_database_state_rejects_inconsistent_audit_totals(monkeypatch):
    fake_connection = FakeConnection(
        [
            (31, 1, 31),
            (31, 30, 0),  # Audit totals do not add up to records_seen.
        ]
    )

    monkeypatch.setattr(
        orchestrator,
        "database_connection",
        lambda: fake_connection,
    )

    staged_rows, audited_rows, matches = orchestrator.get_database_state(
        "fake-checksum",
        2025,
        1,
    )

    assert staged_rows == 31
    assert audited_rows == 31
    assert matches is False
