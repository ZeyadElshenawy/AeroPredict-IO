import csv
import io
import os
import zipfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

zip_path = os.getenv("ZIP_PATH")

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


def main() -> None:
    if not zip_path:
        raise ValueError("ZIP_PATH is not configured. Set it in your .env file.")

    archive_path = Path(zip_path)

    if not archive_path.is_file():
        raise FileNotFoundError(f"ZIP file does not exist: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        csv_files = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(f"Expected one CSV, found {len(csv_files)}")

        with archive.open(csv_files[0]) as binary_file:
            with io.TextIOWrapper(
                binary_file,
                encoding="utf-8-sig",
                newline="",
            ) as text_file:
                reader = csv.DictReader(text_file)

                columns = set(reader.fieldnames or [])
                missing = EXPECTED_COLUMNS - columns

                if missing:
                    raise ValueError(f"Missing expected columns: {sorted(missing)}")

                rows = 0
                months = Counter()
                targets = Counter()
                cancellations = 0
                diversions = 0
                missing_targets = 0

                for row in reader:
                    rows += 1
                    months[(row["YEAR"], row["MONTH"])] += 1

                    target = row["DEP_DEL15"].strip()
                    if target:
                        targets[target] += 1
                    else:
                        missing_targets += 1

                    cancellations += row["CANCELLED"].strip() == "1.00"
                    diversions += row["DIVERTED"].strip() == "1.00"

                print(f"Rows: {rows:,}")
                print(f"Columns: {len(columns)}")
                print(f"Months: {sorted(months)}")
                print(f"Target values: {targets}")
                print(f"Missing targets: {missing_targets:,}")
                print(f"Cancelled rows: {cancellations:,}")
                print(f"Diverted rows: {diversions:,}")


if __name__ == "__main__":
    main()
