import csv
import hashlib
import io
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from flightops_ml.ingestion.loader import (
    SOURCE_COLUMNS,
    database_connection,
    load_month,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MEMBER_NAME = "T_ONTIME_REPORTING.csv"


def make_row(row_number: int) -> dict[str, str]:
    values = {
        "YEAR": "2025.00",
        "QUARTER": "1.00",
        "MONTH": "1.00",
        "DAY_OF_MONTH": "1.00",
        "DAY_OF_WEEK": "3.00",
        "FL_DATE": "1/1/2025 12:00:00 AM",
        "OP_UNIQUE_CARRIER": "ZZ",
        "ORIGIN_AIRPORT_ID": "10001.00",
        "ORIGIN": "AAA",
        "DEST_AIRPORT_ID": "10002.00",
        "DEST": "BBB",
        "CRS_DEP_TIME": "0500",
        "DEP_DEL15": "0.00",
        "CRS_ARR_TIME": "0700",
        "CANCELLED": "0.00",
        "DIVERTED": "0.00",
        "CRS_ELAPSED_TIME": "120.00",
        "DISTANCE": "500.00",
    }
    # Make each synthetic row distinguishable.
    values["ORIGIN_AIRPORT_ID"] = str(10000 + row_number)
    return values


def create_archive(path: Path, row_count: int, malformed_last: bool = False) -> None:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=SOURCE_COLUMNS)
    writer.writeheader()

    for number in range(1, row_count + 1):
        row = make_row(number)
        if malformed_last and number == row_count:
            text.seek(0, io.SEEK_END)
            text.write(",".join(row[col] for col in SOURCE_COLUMNS) + ",bad\n")
        else:
            writer.writerow(row)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MEMBER_NAME, text.getvalue())


def count_staging(checksum: str) -> int:
    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bts_flight_staging WHERE source_sha256 = %s",
                (checksum,),
            )
            return cur.fetchone()[0]


def cleanup(checksums: list[str]) -> None:
    import time

    if not checksums:
        return

    total_start = time.perf_counter()
    start = time.perf_counter()
    conn_context = database_connection()
    conn = conn_context.__enter__()
    print(f"Cleanup connect:          {time.perf_counter() - start:.2f}s")

    try:
        with conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(
                "DELETE FROM public.bts_flight_staging WHERE source_sha256 = ANY(%s)",
                (checksums,),
            )
            print(f"Cleanup staging DELETE:   {time.perf_counter() - start:.2f}s")

            start = time.perf_counter()
            cur.execute(
                "DELETE FROM public.ingestion_runs WHERE source_sha256 = ANY(%s)",
                (checksums,),
            )
            print(f"Cleanup runs DELETE:      {time.perf_counter() - start:.2f}s")

        start = time.perf_counter()
        conn.commit()
        print(f"Cleanup COMMIT:            {time.perf_counter() - start:.2f}s")
    finally:
        start = time.perf_counter()
        conn_context.__exit__(None, None, None)
        print(f"Cleanup context exit:      {time.perf_counter() - start:.2f}s")
        print(f"Cleanup measured total:    {time.perf_counter() - total_start:.2f}s")


def main() -> None:
    checksums: list[str] = []

    with TemporaryDirectory(prefix="flightops-smoke-") as temp_dir:
        temp = Path(temp_dir)
        valid_archive = temp / "synthetic-valid.zip"
        malformed_archive = temp / "synthetic-malformed.zip"

        create_archive(valid_archive, row_count=3)
        create_archive(malformed_archive, row_count=2001, malformed_last=True)

        valid_checksum = hashlib.sha256(valid_archive.read_bytes()).hexdigest()
        malformed_checksum = hashlib.sha256(malformed_archive.read_bytes()).hexdigest()
        checksums.extend([valid_checksum, malformed_checksum])

        try:
            first = load_month(valid_archive, year=2025, month=1)
            assert first["records_seen"] == 3, first
            assert first["records_inserted"] == 3, first
            assert first["records_skipped"] == 0, first
            assert count_staging(valid_checksum) == 3
            print("PASS: first load inserts three records")

            second = load_month(valid_archive, year=2025, month=1)
            assert second["records_seen"] == 3, second
            assert second["records_inserted"] == 0, second
            assert second["records_skipped"] == 3, second
            assert count_staging(valid_checksum) == 3
            print("PASS: repeat load skips duplicate records")

            try:
                load_month(malformed_archive, year=2025, month=1)
            except ValueError as exc:
                assert "Malformed CSV record" in str(exc), str(exc)
            else:
                raise AssertionError("Malformed archive unexpectedly succeeded")

            assert count_staging(malformed_checksum) == 0
            print("PASS: failed load rolls back its partial staging inserts")

        finally:
            cleanup(checksums)

    print("PASS: smoke test complete; synthetic database records cleaned up")


if __name__ == "__main__":
    main()
