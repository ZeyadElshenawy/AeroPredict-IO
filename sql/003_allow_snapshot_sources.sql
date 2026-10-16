ALTER TABLE public.ingestion_runs
    ALTER COLUMN source_year DROP NOT NULL;

ALTER TABLE public.ingestion_runs
    ALTER COLUMN source_month DROP NOT NULL;

ALTER TABLE public.ingestion_runs
    DROP CONSTRAINT ingestion_runs_source_month_check;

ALTER TABLE public.ingestion_runs
    ADD CONSTRAINT ck_ingestion_runs_source_period
    CHECK (
        (source_year IS NULL AND source_month IS NULL)
        OR
        (
            source_year IS NOT NULL
            AND source_month IS NOT NULL
            AND source_year > 0
            AND source_month BETWEEN 1 AND 12
        )
    );