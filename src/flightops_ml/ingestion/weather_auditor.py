"""Read-only weather availability audit engine for FlightOps ML.

Audits actual IEM ASOS/AWOS observation completeness, timestamp cadence,
temporal gaps, and variable missingness across the 2025 calendar year.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_USER_AGENT = "FlightOpsML/0.1.0 (Data Science Research Project)"

CORE_WEATHER_VARIABLES = ("tmpf", "dwpf", "sknt", "drct", "vsby", "alti")

REPRESENTATIVE_PILOT_STATIONS = [
    # 10 existing hubs from January 2025 test dataset
    "ATL",
    "BOS",
    "CLT",
    "DEN",
    "DFW",
    "LAX",
    "MIA",
    "ORD",
    "PHX",
    "SEA",
    # 20 high-volume national hubs across diverse US climate zones
    "JFK",
    "LGA",
    "EWR",
    "SFO",
    "SAN",
    "OAK",
    "MCO",
    "TPA",
    "FLL",
    "BWI",
    "IAD",
    "DCA",
    "DTW",
    "MSP",
    "SLC",
    "PDX",
    "LAS",
    "MDW",
    "BNA",
    "AUS",
    # 10 regional, island territory, and colocated airfield stations
    "PANC",
    "PAFA",
    "PHNL",
    "PHOG",
    "TJSJ",
    "TIST",
    "NSTU",
    "PGSN",
    "IWA",
    "DJT",
]

DAYS_IN_MONTH_2025 = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


@dataclass(frozen=True)
class MonthlyStationAudit:
    station_id: str
    station_name: str
    year: int
    month: int
    month_label: str
    nominal_hours: int
    total_observations: int
    distinct_timestamps: int
    duplicate_timestamps: int
    max_gap_hours: float
    gaps_gt_3h: int
    gaps_gt_6h: int
    null_rate_tmpf: float
    null_rate_dwpf: float
    null_rate_sknt: float
    null_rate_drct: float
    null_rate_vsby: float
    null_rate_alti: float
    completeness_pct: float


@dataclass(frozen=True)
class StationAuditResult:
    station_id: str
    station_name: str
    network: str
    origin_airport_id: int
    airport_code: str
    expected_period: str
    expected_nominal_hours: int
    first_observation_utc: str | None
    last_observation_utc: str | None
    total_observations: int
    distinct_timestamps: int
    duplicate_timestamps: int
    total_gaps_gt_3h: int
    total_gaps_gt_6h: int
    max_gap_hours: float
    null_rate_tmpf: float
    null_rate_dwpf: float
    null_rate_sknt: float
    null_rate_drct: float
    null_rate_vsby: float
    null_rate_alti: float
    status: str  # SUCCEEDED, PARTIAL, FAILED
    requires_investigation: bool
    investigation_reason: str | None
    error_message: str | None


def compute_file_sha256(path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_timestamp_utc(value: str) -> datetime:
    """Parse IEM observation timestamp in UTC format YYYY-MM-DD HH:MM."""
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)


def build_iem_request_params(
    stations: list[str],
    year: int = 2025,
    variables: tuple[str, ...] = CORE_WEATHER_VARIABLES,
    report_types: tuple[str, ...] = ("3", "4"),
) -> dict[str, Any]:
    """Construct query parameters for IEM asos.py CGI endpoint."""
    return {
        "station": stations,
        "data": list(variables),
        "year1": year,
        "month1": 1,
        "day1": 1,
        "hour1": 0,
        "minute1": 0,
        "year2": year,
        "month2": 12,
        "day2": 31,
        "hour2": 23,
        "minute2": 59,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "report_type": list(report_types),
    }


def parse_observation_rows(
    csv_text: str,
    variables: tuple[str, ...] = CORE_WEATHER_VARIABLES,
) -> dict[str, list[dict[str, Any]]]:
    """Parse comma-separated IEM response text into per-station observation lists.

    Preserves station identifiers exactly without altering leading zeros.
    """
    station_rows: dict[str, list[dict[str, Any]]] = {}
    lines = csv_text.strip().splitlines()
    if not lines:
        return station_rows

    reader = csv.DictReader(lines)
    if (
        reader.fieldnames is None
        or "station" not in reader.fieldnames
        or "valid" not in reader.fieldnames
    ):
        return station_rows

    for row in reader:
        raw_stid = row.get("station", "").strip()
        if not raw_stid:
            continue
        # Preserve exact string representation (e.g. '01M')
        stid = raw_stid.upper()
        if stid not in station_rows:
            station_rows[stid] = []

        valid_str = row.get("valid", "").strip()
        try:
            dt = parse_timestamp_utc(valid_str)
        except ValueError:
            continue

        var_values: dict[str, str | None] = {}
        for var in variables:
            val = row.get(var)
            if val is None or not val.strip() or val.strip().upper() == "M":
                var_values[var] = None
            else:
                var_values[var] = val.strip()

        station_rows[stid].append(
            {
                "timestamp": dt,
                "timestamp_str": valid_str,
                **var_values,
            }
        )

    return station_rows


def calculate_station_metrics(
    station_id: str,
    station_name: str,
    network: str,
    origin_airport_id: int,
    airport_code: str,
    observations: list[dict[str, Any]],
    year: int = 2025,
    variables: tuple[str, ...] = CORE_WEATHER_VARIABLES,
) -> tuple[StationAuditResult, list[MonthlyStationAudit]]:
    """Compute annual audit metrics and monthly breakdown for a single station."""
    expected_period = f"{year}-01-01 00:00 to {year}-12-31 23:59 UTC"
    expected_nominal_hours = 8760 if year % 4 != 0 else 8784

    if not observations:
        annual = StationAuditResult(
            station_id=station_id,
            station_name=station_name,
            network=network,
            origin_airport_id=origin_airport_id,
            airport_code=airport_code,
            expected_period=expected_period,
            expected_nominal_hours=expected_nominal_hours,
            first_observation_utc=None,
            last_observation_utc=None,
            total_observations=0,
            distinct_timestamps=0,
            duplicate_timestamps=0,
            total_gaps_gt_3h=0,
            total_gaps_gt_6h=0,
            max_gap_hours=round(expected_nominal_hours, 2),
            null_rate_tmpf=1.0,
            null_rate_dwpf=1.0,
            null_rate_sknt=1.0,
            null_rate_drct=1.0,
            null_rate_vsby=1.0,
            null_rate_alti=1.0,
            status="FAILED",
            requires_investigation=True,
            investigation_reason="Zero observations returned by IEM archive for 2025",
            error_message="No observations found",
        )
        monthly: list[MonthlyStationAudit] = []
        for m in range(1, 13):
            nom_h = DAYS_IN_MONTH_2025[m] * 24
            monthly.append(
                MonthlyStationAudit(
                    station_id=station_id,
                    station_name=station_name,
                    year=year,
                    month=m,
                    month_label=f"{year}-{m:02d}",
                    nominal_hours=nom_h,
                    total_observations=0,
                    distinct_timestamps=0,
                    duplicate_timestamps=0,
                    max_gap_hours=float(nom_h),
                    gaps_gt_3h=1,
                    gaps_gt_6h=1,
                    null_rate_tmpf=1.0,
                    null_rate_dwpf=1.0,
                    null_rate_sknt=1.0,
                    null_rate_drct=1.0,
                    null_rate_vsby=1.0,
                    null_rate_alti=1.0,
                    completeness_pct=0.0,
                )
            )
        return annual, monthly

    # Sort strictly chronologically
    sorted_obs = sorted(observations, key=lambda x: x["timestamp"])
    total_obs = len(sorted_obs)
    timestamps = [o["timestamp"] for o in sorted_obs]
    distinct_ts = len(set(timestamps))
    duplicate_ts = total_obs - distinct_ts

    first_ts = sorted_obs[0]["timestamp_str"]
    last_ts = sorted_obs[-1]["timestamp_str"]

    # Calculate gaps between consecutive distinct observations
    unique_sorted_ts = sorted(set(timestamps))
    gaps_gt_3h = 0
    gaps_gt_6h = 0
    max_gap_hours = 0.0

    # Gap from start of year (2025-01-01 00:00 UTC) to first observation
    year_start = datetime(year, 1, 1, 0, 0, tzinfo=UTC)
    year_end = datetime(year, 12, 31, 23, 59, tzinfo=UTC)

    lead_gap_h = max(0.0, (unique_sorted_ts[0] - year_start).total_seconds() / 3600.0)
    if lead_gap_h > 3.0:
        gaps_gt_3h += 1
    if lead_gap_h > 6.0:
        gaps_gt_6h += 1
    max_gap_hours = max(max_gap_hours, lead_gap_h)

    for i in range(len(unique_sorted_ts) - 1):
        gap_h = (unique_sorted_ts[i + 1] - unique_sorted_ts[i]).total_seconds() / 3600.0
        if gap_h > 3.0:
            gaps_gt_3h += 1
        if gap_h > 6.0:
            gaps_gt_6h += 1
        if gap_h > max_gap_hours:
            max_gap_hours = gap_h

    trail_gap_h = max(0.0, (year_end - unique_sorted_ts[-1]).total_seconds() / 3600.0)
    if trail_gap_h > 3.0:
        gaps_gt_3h += 1
    if trail_gap_h > 6.0:
        gaps_gt_6h += 1
    max_gap_hours = max(max_gap_hours, trail_gap_h)

    # Compute null rates across all observations
    null_counts = {
        var: sum(1 for o in sorted_obs if o.get(var) is None) for var in variables
    }
    null_rates = {var: round(null_counts[var] / total_obs, 4) for var in variables}

    # Assess quality and investigation flags
    requires_investigation = False
    investigation_reasons = []

    if total_obs < 7000:
        requires_investigation = True
        investigation_reasons.append(f"Low observation count: {total_obs:,} < 7,000")

    if max_gap_hours > 24.0:
        requires_investigation = True
        investigation_reasons.append(
            f"Severe telemetry gap: max gap is {max_gap_hours:.1f} hours (>24h)"
        )

    for var in ("tmpf", "sknt", "vsby", "alti"):
        if null_rates.get(var, 0.0) > 0.15:
            requires_investigation = True
            investigation_reasons.append(
                f"Elevated missingness on {var}: {null_rates[var] * 100:.1f}%"
            )

    status = "PARTIAL" if requires_investigation else "SUCCEEDED"
    inv_reason_str = "; ".join(investigation_reasons) if investigation_reasons else None

    annual_result = StationAuditResult(
        station_id=station_id,
        station_name=station_name,
        network=network,
        origin_airport_id=origin_airport_id,
        airport_code=airport_code,
        expected_period=expected_period,
        expected_nominal_hours=expected_nominal_hours,
        first_observation_utc=first_ts,
        last_observation_utc=last_ts,
        total_observations=total_obs,
        distinct_timestamps=distinct_ts,
        duplicate_timestamps=duplicate_ts,
        total_gaps_gt_3h=gaps_gt_3h,
        total_gaps_gt_6h=gaps_gt_6h,
        max_gap_hours=round(max_gap_hours, 2),
        null_rate_tmpf=null_rates.get("tmpf", 0.0),
        null_rate_dwpf=null_rates.get("dwpf", 0.0),
        null_rate_sknt=null_rates.get("sknt", 0.0),
        null_rate_drct=null_rates.get("drct", 0.0),
        null_rate_vsby=null_rates.get("vsby", 0.0),
        null_rate_alti=null_rates.get("alti", 0.0),
        status=status,
        requires_investigation=requires_investigation,
        investigation_reason=inv_reason_str,
        error_message=None,
    )

    # Compute monthly breakdowns
    monthly_results: list[MonthlyStationAudit] = []
    obs_by_month: dict[int, list[dict[str, Any]]] = {m: [] for m in range(1, 13)}
    for o in sorted_obs:
        m = o["timestamp"].month
        obs_by_month[m].append(o)

    for m in range(1, 13):
        m_obs = obs_by_month[m]
        nom_h = DAYS_IN_MONTH_2025[m] * 24
        m_total = len(m_obs)
        if not m_obs:
            monthly_results.append(
                MonthlyStationAudit(
                    station_id=station_id,
                    station_name=station_name,
                    year=year,
                    month=m,
                    month_label=f"{year}-{m:02d}",
                    nominal_hours=nom_h,
                    total_observations=0,
                    distinct_timestamps=0,
                    duplicate_timestamps=0,
                    max_gap_hours=float(nom_h),
                    gaps_gt_3h=1,
                    gaps_gt_6h=1,
                    null_rate_tmpf=1.0,
                    null_rate_dwpf=1.0,
                    null_rate_sknt=1.0,
                    null_rate_drct=1.0,
                    null_rate_vsby=1.0,
                    null_rate_alti=1.0,
                    completeness_pct=0.0,
                )
            )
            continue

        m_ts = [o["timestamp"] for o in m_obs]
        m_distinct = len(set(m_ts))
        m_dupes = m_total - m_distinct
        m_u_ts = sorted(set(m_ts))

        m_max_gap = 0.0
        m_gaps_gt_3 = 0
        m_gaps_gt_6 = 0
        for i in range(len(m_u_ts) - 1):
            g = (m_u_ts[i + 1] - m_u_ts[i]).total_seconds() / 3600.0
            if g > 3.0:
                m_gaps_gt_3 += 1
            if g > 6.0:
                m_gaps_gt_6 += 1
            if g > m_max_gap:
                m_max_gap = g

        m_null_rates = {
            var: round(sum(1 for o in m_obs if o.get(var) is None) / m_total, 4)
            for var in variables
        }
        comp_pct = round(min(100.0, (m_distinct / nom_h) * 100.0), 2)

        monthly_results.append(
            MonthlyStationAudit(
                station_id=station_id,
                station_name=station_name,
                year=year,
                month=m,
                month_label=f"{year}-{m:02d}",
                nominal_hours=nom_h,
                total_observations=m_total,
                distinct_timestamps=m_distinct,
                duplicate_timestamps=m_dupes,
                max_gap_hours=round(m_max_gap, 2),
                gaps_gt_3h=m_gaps_gt_3,
                gaps_gt_6h=m_gaps_gt_6,
                null_rate_tmpf=m_null_rates.get("tmpf", 0.0),
                null_rate_dwpf=m_null_rates.get("dwpf", 0.0),
                null_rate_sknt=m_null_rates.get("sknt", 0.0),
                null_rate_drct=m_null_rates.get("drct", 0.0),
                null_rate_vsby=m_null_rates.get("vsby", 0.0),
                null_rate_alti=m_null_rates.get("alti", 0.0),
                completeness_pct=comp_pct,
            )
        )

    return annual_result, monthly_results


def fetch_iem_batch(
    session: requests.Session,
    stations: list[str],
    year: int = 2025,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: float = 45.0,
) -> str:
    """Fetch observation stream from IEM asos.py with exponential retry backoff."""
    params = build_iem_request_params(stations, year=year)
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(
                IEM_ASOS_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code in (429, 503):
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch batch for stations {stations} after "
                        f"{max_retries} attempts: HTTP {resp.status_code}"
                    )
                # Rate limited or server busy
                wait_time = backoff_factor**attempt
                time.sleep(wait_time)
                continue
            else:
                resp.raise_for_status()

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_factor**attempt)
            else:
                raise RuntimeError(
                    f"Failed to fetch batch for stations {stations} after "
                    f"{max_retries} attempts: {exc}"
                ) from exc

    if last_exc:
        raise last_exc
    raise RuntimeError(
        f"Failed to fetch batch for stations {stations} after {max_retries} attempts"
    )


def execute_weather_audit(
    station_mappings: list[dict[str, Any]],
    batch_size: int = 5,
    request_pause_seconds: float = 1.5,
    cache_path: Path | None = None,
    session: requests.Session | None = None,
) -> tuple[list[StationAuditResult], list[MonthlyStationAudit]]:
    """Execute audited observation extraction across a collection of mapped stations."""
    if session is None:
        managed_session = requests.Session()
    else:
        managed_session = session

    # Load cache if available for resumability
    completed_stations: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                completed_stations = json.load(f)
        except Exception:
            completed_stations = {}

    annual_results: list[StationAuditResult] = []
    monthly_results: list[MonthlyStationAudit] = []

    # Map station ID to metadata
    meta_lookup = {r["station_id"]: r for r in station_mappings}
    all_stids = [r["station_id"] for r in station_mappings]

    # Process in batches
    batches = [
        all_stids[i : i + batch_size] for i in range(0, len(all_stids), batch_size)
    ]

    for batch_num, batch_stids in enumerate(batches, start=1):
        pending = [s for s in batch_stids if s not in completed_stations]
        if not pending:
            # Reconstruct from cache
            for s in batch_stids:
                cached = completed_stations[s]
                annual_results.append(StationAuditResult(**cached["annual"]))
                for m_dict in cached["monthly"]:
                    monthly_results.append(MonthlyStationAudit(**m_dict))
            continue

        try:
            csv_text = fetch_iem_batch(managed_session, pending, year=2025)
            parsed_rows = parse_observation_rows(csv_text)

            for stid in pending:
                meta = meta_lookup[stid]
                obs = parsed_rows.get(stid, [])
                ann, mon = calculate_station_metrics(
                    station_id=stid,
                    station_name=meta.get("station_name", ""),
                    network=meta.get("network", ""),
                    origin_airport_id=int(meta.get("origin_airport_id", 0)),
                    airport_code=meta.get("airport_code", ""),
                    observations=obs,
                )
                annual_results.append(ann)
                monthly_results.extend(mon)

                completed_stations[stid] = {
                    "annual": asdict(ann),
                    "monthly": [asdict(m) for m in mon],
                }

        except Exception as exc:
            # Record failed stations gracefully
            for stid in pending:
                meta = meta_lookup[stid]
                fail_res = StationAuditResult(
                    station_id=stid,
                    station_name=meta.get("station_name", ""),
                    network=meta.get("network", ""),
                    origin_airport_id=int(meta.get("origin_airport_id", 0)),
                    airport_code=meta.get("airport_code", ""),
                    expected_period="2025-01-01 00:00 to 2025-12-31 23:59 UTC",
                    expected_nominal_hours=8760,
                    first_observation_utc=None,
                    last_observation_utc=None,
                    total_observations=0,
                    distinct_timestamps=0,
                    duplicate_timestamps=0,
                    total_gaps_gt_3h=0,
                    total_gaps_gt_6h=0,
                    max_gap_hours=8760.0,
                    null_rate_tmpf=1.0,
                    null_rate_dwpf=1.0,
                    null_rate_sknt=1.0,
                    null_rate_drct=1.0,
                    null_rate_vsby=1.0,
                    null_rate_alti=1.0,
                    status="FAILED",
                    requires_investigation=True,
                    investigation_reason=f"HTTP / Network failure: {exc}",
                    error_message=str(exc)[:500],
                )
                annual_results.append(fail_res)
                # Also generate empty monthly records
                for m in range(1, 13):
                    nom_h = DAYS_IN_MONTH_2025[m] * 24
                    monthly_results.append(
                        MonthlyStationAudit(
                            station_id=stid,
                            station_name=meta.get("station_name", ""),
                            year=2025,
                            month=m,
                            month_label=f"2025-{m:02d}",
                            nominal_hours=nom_h,
                            total_observations=0,
                            distinct_timestamps=0,
                            duplicate_timestamps=0,
                            max_gap_hours=float(nom_h),
                            gaps_gt_3h=1,
                            gaps_gt_6h=1,
                            null_rate_tmpf=1.0,
                            null_rate_dwpf=1.0,
                            null_rate_sknt=1.0,
                            null_rate_drct=1.0,
                            null_rate_vsby=1.0,
                            null_rate_alti=1.0,
                            completeness_pct=0.0,
                        )
                    )
                completed_stations[stid] = {
                    "annual": asdict(fail_res),
                    "monthly": [asdict(m) for m in monthly_results[-12:]],
                }

        # Update cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(completed_stations, f, indent=2)

        # Respect 1-second throttle
        if batch_num < len(batches):
            time.sleep(request_pause_seconds)

    return annual_results, monthly_results


def export_audit_artifacts(
    annual_results: list[StationAuditResult],
    monthly_results: list[MonthlyStationAudit],
    output_dir: Path,
    scope: str = "pilot",
    catalogue_path: Path | None = None,
    mapping_path: Path | None = None,
) -> dict[str, Any]:
    """Export audit results to annual CSV, monthly CSV, and summary JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annual_csv_path = output_dir / f"weather_availability_annual_{scope}.csv"
    with annual_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(annual_results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in annual_results:
            writer.writerow(asdict(res))

    monthly_csv_path = output_dir / f"weather_availability_monthly_{scope}.csv"
    with monthly_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(monthly_results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m_res in monthly_results:
            writer.writerow(asdict(m_res))

    # Calculate summary metrics
    total_audited = len(annual_results)
    succeeded = sum(1 for r in annual_results if r.status == "SUCCEEDED")
    partial = sum(1 for r in annual_results if r.status == "PARTIAL")
    failed = sum(1 for r in annual_results if r.status == "FAILED")
    inv_required = sum(1 for r in annual_results if r.requires_investigation)

    total_obs = sum(r.total_observations for r in annual_results)
    avg_obs = round(total_obs / total_audited, 1) if total_audited else 0.0

    sorted_gaps = sorted(r.max_gap_hours for r in annual_results)
    med_max_gap = sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else 0.0

    avg_nulls = {
        var: round(
            sum(getattr(r, f"null_rate_{var}") for r in annual_results) / total_audited,
            4,
        )
        for var in CORE_WEATHER_VARIABLES
    }

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_scope": scope,
        "target_period": "2025-01-01 00:00 to 2025-12-31 23:59 UTC",
        "nominal_hours_per_station": 8760,
        "stations_audited": total_audited,
        "stations_succeeded": succeeded,
        "stations_partial": partial,
        "stations_failed": failed,
        "stations_requiring_investigation": inv_required,
        "investigation_stations": [
            {
                "station_id": r.station_id,
                "station_name": r.station_name,
                "total_observations": r.total_observations,
                "max_gap_hours": r.max_gap_hours,
                "reason": r.investigation_reason,
            }
            for r in annual_results
            if r.requires_investigation
        ],
        "aggregate_observations": total_obs,
        "average_observations_per_station": avg_obs,
        "median_max_gap_hours": med_max_gap,
        "average_null_rates": avg_nulls,
        "catalogue_sha256": (
            compute_file_sha256(catalogue_path)
            if catalogue_path and catalogue_path.exists()
            else None
        ),
        "mapping_sha256": (
            compute_file_sha256(mapping_path)
            if mapping_path and mapping_path.exists()
            else None
        ),
        "files_generated": [
            str(annual_csv_path),
            str(monthly_csv_path),
            str(output_dir / f"weather_availability_summary_{scope}.json"),
        ],
    }

    summary_json_path = output_dir / f"weather_availability_summary_{scope}.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
