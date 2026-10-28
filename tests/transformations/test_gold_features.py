import math
from datetime import date, time
from unittest.mock import MagicMock, patch

import pytest
from scripts import transform_gold_features as orchestrator

from flightops_ml.transformations import gold_features as gf


def test_assign_dataset_split_boundaries():
    # Train boundaries
    assert gf.assign_dataset_split(date(2025, 1, 1)) == "train"
    assert gf.assign_dataset_split(date(2025, 8, 31)) == "train"

    # Validation boundaries
    assert gf.assign_dataset_split(date(2025, 9, 1)) == "validation"
    assert gf.assign_dataset_split(date(2025, 10, 31)) == "validation"

    # Test boundaries
    assert gf.assign_dataset_split(date(2025, 11, 1)) == "test"
    assert gf.assign_dataset_split(date(2025, 12, 31)) == "test"

    # Out of bounds
    with pytest.raises(ValueError, match="precedes 2025 train boundary"):
        gf.assign_dataset_split(date(2024, 12, 31))
    with pytest.raises(ValueError, match="exceeds 2025 test boundary"):
        gf.assign_dataset_split(date(2026, 1, 1))


def test_split_boundaries_are_contiguous_and_disjoint():
    """Verify split date ranges do not overlap and leave no gap."""
    assert gf.TRAIN_START_DATE == date(2025, 1, 1)
    assert gf.TRAIN_END_DATE == date(2025, 8, 31)
    assert gf.VAL_START_DATE == date(2025, 9, 1)
    assert gf.VAL_END_DATE == date(2025, 10, 31)
    assert gf.TEST_START_DATE == date(2025, 11, 1)
    assert gf.TEST_END_DATE == date(2025, 12, 31)

    # Contiguity
    assert (gf.VAL_START_DATE - gf.TRAIN_END_DATE).days == 1
    assert (gf.TEST_START_DATE - gf.VAL_END_DATE).days == 1


def test_compute_cyclical_dep_time():
    # Midnight (00:00)
    h, m, sin_t, cos_t = gf.compute_cyclical_dep_time(time(0, 0))
    assert (h, m) == (0, 0)
    assert math.isclose(sin_t, 0.0, abs_tol=1e-5)
    assert math.isclose(cos_t, 1.0, abs_tol=1e-5)

    # 06:00 (quarter day)
    h, m, sin_t, cos_t = gf.compute_cyclical_dep_time(time(6, 0))
    assert (h, m) == (6, 0)
    assert math.isclose(sin_t, 1.0, abs_tol=1e-5)
    assert math.isclose(cos_t, 0.0, abs_tol=1e-5)

    # 12:00 (half day)
    h, m, sin_t, cos_t = gf.compute_cyclical_dep_time(time(12, 0))
    assert (h, m) == (12, 0)
    assert math.isclose(sin_t, 0.0, abs_tol=1e-5)
    assert math.isclose(cos_t, -1.0, abs_tol=1e-5)

    # 18:00 (three quarter day)
    h, m, sin_t, cos_t = gf.compute_cyclical_dep_time(time(18, 0))
    assert (h, m) == (18, 0)
    assert math.isclose(sin_t, -1.0, abs_tol=1e-5)
    assert math.isclose(cos_t, 0.0, abs_tol=1e-5)

    # BTS 2400 string convention
    h, m, sin_t, cos_t = gf.compute_cyclical_dep_time("2400")
    assert (h, m) == (0, 0)
    assert math.isclose(sin_t, 0.0, abs_tol=1e-5)
    assert math.isclose(cos_t, 1.0, abs_tol=1e-5)

    with pytest.raises(ValueError, match="Invalid hour/minute"):
        gf.compute_cyclical_dep_time("2500")


def test_compute_cyclical_day_of_week_and_month():
    # Day of week
    sin_d, cos_d = gf.compute_cyclical_day_of_week(1)
    assert math.isclose(sin_d, 0.0, abs_tol=1e-5)
    assert math.isclose(cos_d, 1.0, abs_tol=1e-5)

    sin_m, cos_m = gf.compute_cyclical_month(1)
    assert math.isclose(sin_m, 0.0, abs_tol=1e-5)
    assert math.isclose(cos_m, 1.0, abs_tol=1e-5)

    with pytest.raises(ValueError, match="day_of_week"):
        gf.compute_cyclical_day_of_week(8)
    with pytest.raises(ValueError, match="month"):
        gf.compute_cyclical_month(13)


def test_feature_allowlist_enforcement():
    # Valid allowlist passes
    gf.validate_feature_allowlist(gf.PREDICTOR_ALLOWLIST)

    # Target leakage fails
    with pytest.raises(ValueError, match="Target leakage detected"):
        gf.validate_feature_allowlist(["month", "dep_del15"])

    # Prohibited actual/outcome terms fail
    with pytest.raises(ValueError, match="Forbidden leakage term 'actual'"):
        gf.validate_feature_allowlist(["month", "actual_dep_time"])

    with pytest.raises(ValueError, match="Forbidden leakage term 'dep_delay'"):
        gf.validate_feature_allowlist(["month", "dep_delay"])

    with pytest.raises(ValueError, match="Forbidden leakage term 'weather'"):
        gf.validate_feature_allowlist(["month", "weather_condition"])

    with pytest.raises(ValueError, match="Forbidden leakage term 'variation'"):
        gf.validate_feature_allowlist(["month", "utc_local_time_variation"])

    with pytest.raises(ValueError, match="Forbidden leakage term 'impute'"):
        gf.validate_feature_allowlist(["month", "crs_elapsed_time_imputed"])

    # Unallowed feature fails
    with pytest.raises(ValueError, match="Unallowed features detected"):
        gf.validate_feature_allowlist(["month", "arbitrary_unregistered_feature"])


def test_transform_gold_row_success():
    silver_row = {
        "source_sha256": "f" * 64,
        "source_row_number": 42,
        "flight_date": date(2025, 7, 24),
        "is_training_eligible": True,
        "dep_del15": 1,
        "year": 2025,
        "month": 7,
        "day_of_month": 24,
        "day_of_week": 4,
        "quarter": 3,
        "crs_dep_time": time(9, 22),
        "op_unique_carrier": "DL",
        "origin_airport_id": 10397,
        "dest_airport_id": 12892,
        "distance": 1947.0,
        "crs_elapsed_time": 255.0,
    }
    orig_coords = (33.6407, -84.4277)
    dest_coords = (33.9425, -118.4081)

    gold = gf.transform_gold_row(silver_row, orig_coords, dest_coords)

    assert gold["dataset_split"] == "train"
    assert gold["dep_del15"] == 1
    assert gold["dep_hour"] == 9
    assert gold["dep_minute"] == 22
    assert gold["is_weekend"] is False
    assert gold["crs_elapsed_time_clean"] == 255.0
    assert "crs_elapsed_time_imputed" not in gold
    assert gold["origin_latitude"] == 33.6407
    assert gold["dest_longitude"] == -118.4081

    # Ensure source record was not modified in-place
    assert silver_row["crs_elapsed_time"] == 255.0
    assert silver_row["source_row_number"] == 42


def test_transform_gold_row_duration_exclusions():
    """Verify that null, zero, and negative scheduled elapsed times are rejected."""
    base_row = {
        "source_sha256": "f" * 64,
        "source_row_number": 1,
        "flight_date": date(2025, 5, 10),
        "is_training_eligible": True,
        "dep_del15": 0,
        "year": 2025,
        "month": 5,
        "day_of_month": 10,
        "day_of_week": 6,
        "quarter": 2,
        "crs_dep_time": time(10, 0),
        "op_unique_carrier": "AA",
        "origin_airport_id": 10397,
        "dest_airport_id": 12892,
        "distance": 500.0,
    }
    orig_coords = (33.6407, -84.4277)
    dest_coords = (33.9425, -118.4081)

    # 1. Null duration
    row_null = {**base_row, "crs_elapsed_time": None}
    with pytest.raises(ValueError, match="Invalid scheduled duration for Gold"):
        gf.transform_gold_row(row_null, orig_coords, dest_coords)

    # 2. Zero duration
    row_zero = {**base_row, "crs_elapsed_time": 0.0}
    with pytest.raises(ValueError, match="Invalid scheduled duration for Gold"):
        gf.transform_gold_row(row_zero, orig_coords, dest_coords)

    # 3. Negative duration
    row_neg = {**base_row, "crs_elapsed_time": -60.0}
    with pytest.raises(ValueError, match="Invalid scheduled duration for Gold"):
        gf.transform_gold_row(row_neg, orig_coords, dest_coords)

    # 4. Valid positive duration passes unchanged
    row_valid = {**base_row, "crs_elapsed_time": 75.5}
    gold = gf.transform_gold_row(row_valid, orig_coords, dest_coords)
    assert gold["crs_elapsed_time_clean"] == 75.5


def test_transform_gold_row_ineligible_raises():
    silver_row = {
        "source_sha256": "f" * 64,
        "source_row_number": 1,
        "flight_date": date(2025, 7, 24),
        "is_training_eligible": False,  # Cancelled or diverted!
        "dep_del15": None,
        "crs_elapsed_time": 120.0,
    }
    with pytest.raises(ValueError, match="ineligible flight into Gold"):
        gf.transform_gold_row(silver_row, (0.0, 0.0), (0.0, 0.0))


def test_reconciliation_math_across_splits():
    """Verify expected Gold counts reconcile with eligible Silver minus exclusions."""
    audit_data = [
        {
            "split": "train",
            "eligible": 4592366,
            "missing_dur": 0,
            "nonpos_dur": 0,
            "expected_gold": 4592366,
        },
        {
            "split": "validation",
            "eligible": 1159899,
            "missing_dur": 0,
            "nonpos_dur": 0,
            "expected_gold": 1159899,
        },
        {
            "split": "test",
            "eligible": 1127220,
            "missing_dur": 0,
            "nonpos_dur": 1,
            "expected_gold": 1127219,
        },
    ]

    total_eligible = sum(d["eligible"] for d in audit_data)
    total_excluded = sum(d["missing_dur"] + d["nonpos_dur"] for d in audit_data)
    total_expected = sum(d["expected_gold"] for d in audit_data)

    assert total_eligible == 6879485
    assert total_excluded == 1
    assert total_expected == 6879484

    for d in audit_data:
        excluded = d["missing_dur"] + d["nonpos_dur"]
        assert d["eligible"] - excluded == d["expected_gold"]


def test_airport_reference_join_no_fanout_mock():
    """Verify reference table interval match matches exactly one valid interval."""
    reference_records = [
        {
            "airport_id": 14027,
            "airport_code": "PBI",
            "start_date": date(2011, 7, 1),
            "thru_date": date(2026, 7, 31),
            "lat": 26.6832,
            "lon": -80.0956,
        },
        {
            "airport_id": 14027,
            "airport_code": "DJT",
            "start_date": date(2026, 8, 1),
            "thru_date": None,
            "lat": 26.6832,
            "lon": -80.0956,
        },
    ]

    flight_date_2025 = date(2025, 5, 10)

    # Filter by airport_id and flight_date within [start_date, thru_date]
    matches = [
        rec
        for rec in reference_records
        if rec["airport_id"] == 14027
        and rec["start_date"] <= flight_date_2025
        and (rec["thru_date"] is None or flight_date_2025 <= rec["thru_date"])
    ]

    # Must match exactly one record
    assert len(matches) == 1
    assert matches[0]["airport_code"] == "PBI"


def test_cli_dry_run_does_not_modify_database():
    """Verify that orchestrator dry run only reads and never commits writes."""
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value

    fake_cur.fetchone.side_effect = [
        (False,),  # check_gold_table_exists (run_dry_run)
        (False,),  # check_gold_table_exists (get_actual_gold_counts)
        # airport join verification
        (6879484, 6879484, 6879484),
    ]
    fake_cur.fetchall.side_effect = [
        # audit_silver_and_exclusions
        [
            ("train", date(2025, 1, 1), date(2025, 8, 31), 4592366, 0, 0, 4592366),
            (
                "validation",
                date(2025, 9, 1),
                date(2025, 10, 31),
                1159899,
                0,
                0,
                1159899,
            ),
            ("test", date(2025, 11, 1), date(2025, 12, 31), 1127220, 0, 1, 1127219),
        ],
    ]

    with patch("scripts.transform_gold_features.database_connection") as mock_db_conn:
        mock_db_conn.return_value.__enter__.return_value = fake_conn

        # Run CLI in dry-run mode
        ret = orchestrator.main([])
        assert ret == 0

        # Assert no commit was called on the connection
        fake_conn.commit.assert_not_called()


def test_execute_transformation_skips_when_already_complete():
    """Verify that rerun skips insert when Gold table is already complete."""
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value

    fake_cur.fetchone.side_effect = [
        (True,),  # check_gold_table_exists
        (6879484,),  # SELECT COUNT(*) FROM public.gold_features
    ]
    # silver_eligible_vs_gold_counts check
    fake_cur.fetchall.return_value = [
        ("train", 4592366, 0, 0, 4592366, 4592366, True),
        ("validation", 1159899, 0, 0, 1159899, 1159899, True),
        ("test", 1127220, 0, 1, 1127219, 1127219, True),
    ]

    ret = orchestrator.execute_transformation(fake_conn, replace=False)
    assert ret == 0

    # Ensure no INSERT or TRUNCATE was executed
    for call_args in fake_cur.execute.call_args_list:
        query = call_args[0][0]
        assert "INSERT INTO" not in query
        assert "TRUNCATE" not in query


def test_execute_transformation_with_replace_truncates():
    """Verify that --replace executes TRUNCATE before inserting."""
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    fake_cur.rowcount = 6879484

    fake_cur.fetchone.side_effect = [
        (True,),  # check_gold_table_exists
        (6879484,),  # existing count
    ]
    # Post-insert reconciliation check
    fake_cur.fetchall.return_value = [
        ("train", 4592366, 0, 0, 4592366, 4592366, True),
        ("validation", 1159899, 0, 0, 1159899, 1159899, True),
        ("test", 1127220, 0, 1, 1127219, 1127219, True),
    ]

    ret = orchestrator.execute_transformation(fake_conn, replace=True)
    assert ret == 0

    executed_queries = [call[0][0] for call in fake_cur.execute.call_args_list]
    assert any("TRUNCATE TABLE public.gold_features;" in q for q in executed_queries)
    assert any("INSERT INTO public.gold_features" in q for q in executed_queries)
    fake_conn.commit.assert_called_once()


def test_execute_transformation_fails_reconciliation_rolls_back():
    """Verify that reconciliation discrepancy raises error and never commits."""
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    fake_cur.rowcount = 1000

    fake_cur.fetchone.side_effect = [
        (True,),  # check_gold_table_exists
        (0,),  # existing count
    ]
    # Reconciliation failure
    fake_cur.fetchall.return_value = [
        ("train", 4592366, 0, 0, 4592366, 1000, False),
    ]

    with pytest.raises(RuntimeError, match="Gold reconciliation failed"):
        orchestrator.execute_transformation(fake_conn, replace=False)

    fake_conn.commit.assert_not_called()
