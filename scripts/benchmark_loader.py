import hashlib
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke_test_loader import cleanup, count_staging, create_archive

from flightops_ml.ingestion.loader import load_month

ROW_COUNT = 10_000


def main() -> None:
    with TemporaryDirectory(prefix="flightops-benchmark-") as temp_dir:
        archive_path = Path(temp_dir) / "synthetic-benchmark.zip"
        create_archive(archive_path, row_count=ROW_COUNT)
        checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

        try:
            start = time.perf_counter()
            result = load_month(archive_path, year=2025, month=1)
            load_seconds = time.perf_counter() - start

            stored_rows = count_staging(checksum)

            print(f"Rows in source:       {result['records_seen']:,}")
            print(f"Rows inserted:        {result['records_inserted']:,}")
            print(f"Rows skipped:         {result['records_skipped']:,}")
            print(f"Rows verified in DB:  {stored_rows:,}")
            print(f"Ingestion time:       {load_seconds:.2f} seconds")

            if load_seconds > 0:
                print(
                    f"Ingestion throughput: "
                    f"{result['records_seen'] / load_seconds:,.0f} rows/sec"
                )

            assert result["records_seen"] == ROW_COUNT
            assert result["records_inserted"] == ROW_COUNT
            assert result["records_skipped"] == 0
            assert stored_rows == ROW_COUNT

        finally:
            cleanup_start = time.perf_counter()
            cleanup([checksum])
            cleanup_seconds = time.perf_counter() - cleanup_start
            print(f"Cleanup time:         {cleanup_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
