import time
import uuid
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ROW_COUNT = 10_000
BATCH_SIZE = 2_000

COLUMNS = (
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
    "year_raw",
    "quarter_raw",
    "month_raw",
    "day_of_month_raw",
    "day_of_week_raw",
    "flight_date_raw",
    "op_unique_carrier_raw",
    "origin_airport_id_raw",
    "origin_raw",
    "dest_airport_id_raw",
    "dest_raw",
    "crs_dep_time_raw",
    "dep_del15_raw",
    "crs_arr_time_raw",
    "cancelled_raw",
    "diverted_raw",
    "crs_elapsed_time_raw",
    "distance_raw",
)


def connect():
    return psycopg.connect(
        host=__import__("os").getenv("PGHOST", "localhost"),
        port=int(__import__("os").getenv("PGPORT", "5433")),
        dbname=__import__("os").environ["POSTGRES_DB"],
        user=__import__("os").environ["POSTGRES_USER"],
        password=__import__("os").environ["POSTGRES_PASSWORD"],
    )


def synthetic_row(checksum: str, number: int) -> tuple:
    return (
        checksum,
        number,
        1,
        "2025.00",
        "1.00",
        "1.00",
        "1.00",
        "3.00",
        "1/1/2025 12:00:00 AM",
        "ZZ",
        str(10000 + number),
        "AAA",
        str(20000 + number),
        "BBB",
        "0500",
        "0.00",
        "0700",
        "0.00",
        "0.00",
        "120.00",
        "500.00",
    )


def main():
    table = f"benchmark_{uuid.uuid4().hex[:12]}"
    checksum = uuid.uuid4().hex * 2
    quoted_table = f'"{table}"'

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE {quoted_table} (
                    LIKE public.bts_flight_staging INCLUDING ALL
                )
                """
            )
            cur.execute(f"CREATE INDEX ON {quoted_table} (flight_date_raw)")

        try:
            copy_start = time.perf_counter()
            with conn.cursor() as cur:
                with cur.copy(
                    f"""
                    COPY {quoted_table} ({", ".join(COLUMNS)})
                    FROM STDIN
                    """
                ) as copy:
                    for number in range(1, ROW_COUNT + 1):
                        copy.write_row(synthetic_row(checksum, number))
            conn.commit()
            copy_seconds = time.perf_counter() - copy_start

            print(f"COPY inserted: {ROW_COUNT:,} rows")
            print(f"COPY time:     {copy_seconds:.2f} seconds")
            print(f"COPY rate:     {ROW_COUNT / copy_seconds:,.0f} rows/sec")

            batch_start = time.perf_counter()
            batch_checksum = uuid.uuid4().hex * 2
            insert_sql = f"""
                INSERT INTO {quoted_table} ({", ".join(COLUMNS)})
                VALUES ({", ".join(["%s"] * len(COLUMNS))})
            """

            with conn.cursor() as cur:
                for start in range(1, ROW_COUNT + 1, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, ROW_COUNT + 1)
                    cur.executemany(
                        insert_sql,
                        [
                            synthetic_row(batch_checksum, number)
                            for number in range(start, end)
                        ],
                    )
            conn.commit()
            batch_seconds = time.perf_counter() - batch_start

            print(f"Batch inserted: {ROW_COUNT:,} rows")
            print(f"Batch time:     {batch_seconds:.2f} seconds")
            print(f"Batch rate:     {ROW_COUNT / batch_seconds:,.0f} rows/sec")

            delete_start = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {quoted_table}")
            conn.commit()
            delete_seconds = time.perf_counter() - delete_start
            print(f"Delete time:    {delete_seconds:.2f} seconds")

        finally:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            conn.commit()
            print("Temporary benchmark table dropped.")


if __name__ == "__main__":
    main()
