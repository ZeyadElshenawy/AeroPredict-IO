import os
import tempfile
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

from flightops_ml.ingestion.archive import validate_bts_archive

BTS_URL = (
    "https://www.transtats.bts.gov/"
    "DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ"
)

REQUIRED_COLUMNS = (
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN",
    "DEST_AIRPORT_ID",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
)


class HiddenFieldParser(HTMLParser):
    """Extract hidden input fields from an ASP.NET form."""

    def __init__(self):
        super().__init__()
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return

        attributes = dict(attrs)
        if attributes.get("type", "").lower() != "hidden":
            return

        name = attributes.get("name")
        if name:
            self.fields[name] = attributes.get("value", "")


def fetch_form_fields(session: requests.Session) -> dict[str, str]:
    """Fetch the BTS reporting page and extract its hidden form fields."""
    response = session.get(BTS_URL, timeout=30)
    response.raise_for_status()

    parser = HiddenFieldParser()
    parser.feed(response.text)

    required_fields = {
        "__VIEWSTATE",
        "__VIEWSTATEGENERATOR",
        "__EVENTVALIDATION",
    }
    missing_fields = required_fields - parser.fields.keys()

    if missing_fields:
        raise ValueError(
            f"BTS form is missing required hidden fields: {sorted(missing_fields)}"
        )

    return parser.fields


def build_download_payload(
    hidden_fields: dict[str, str],
    year: int,
    month: int,
) -> dict[str, str]:
    """Build and validate a BTS download form payload."""
    if not EARLIEST_YEAR <= year <= LATEST_YEAR:
        raise ValueError(f"year must be between {EARLIEST_YEAR} and {LATEST_YEAR}")

    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    required_hidden_fields = {
        "__VIEWSTATE",
        "__VIEWSTATEGENERATOR",
        "__EVENTVALIDATION",
    }
    missing_fields = required_hidden_fields - hidden_fields.keys()

    if missing_fields:
        raise ValueError(f"Missing required hidden fields: {sorted(missing_fields)}")

    payload = dict(hidden_fields)
    payload.update(
        {
            "cboGeography": "All",
            "cboYear": str(year),
            "cboPeriod": str(month),
            "btnDownload": "Download",
        }
    )

    for column in REQUIRED_COLUMNS:
        payload[column] = "on"

    return payload


class FormControlParser(HTMLParser):
    """Extract dropdown options and input controls from the BTS form."""

    def __init__(self):
        super().__init__()
        self.select_name = None
        self.select_options = {}
        self.inputs = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)

        if tag.lower() == "select":
            self.select_name = attributes.get("name")
            if self.select_name:
                self.select_options[self.select_name] = []

        elif tag.lower() == "option" and self.select_name:
            value = attributes.get("value")
            if value is not None:
                self.select_options[self.select_name].append(value)

        elif tag.lower() == "input":
            name = attributes.get("name")
            if name:
                self.inputs[name] = {
                    "type": attributes.get("type", "").lower(),
                    "value": attributes.get("value", ""),
                    "checked": "checked" in attributes,
                }

    def handle_endtag(self, tag):
        if tag.lower() == "select":
            self.select_name = None


class FormActionParser(HTMLParser):
    """Find the outer BTS form containing the download controls."""

    def __init__(self):
        super().__init__()
        self.form_stack = []
        self.download_form_action = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        tag = tag.lower()

        if tag == "form":
            self.form_stack.append(
                {
                    "action": attributes.get("action", ""),
                    "has_download_controls": False,
                }
            )
            return

        if not self.form_stack:
            return

        if tag == "input" and attributes.get("name") == "btnDownload":
            self.form_stack[-1]["has_download_controls"] = True

        elif tag == "select" and attributes.get("name") in {
            "cboYear",
            "cboPeriod",
        }:
            self.form_stack[-1]["has_download_controls"] = True

    def handle_endtag(self, tag):
        if tag.lower() != "form" or not self.form_stack:
            return

        closed_form = self.form_stack.pop()

        if closed_form["has_download_controls"]:
            self.download_form_action = unescape(closed_form["action"])
        elif self.form_stack:
            # Nested forms are invalid HTML. Propagate the signal so the
            # outer BTS form can still be identified.
            self.form_stack[-1]["has_download_controls"] = True


def fetch_download_form(
    session: requests.Session,
) -> tuple[dict[str, str], str]:
    """Fetch BTS once and return its hidden fields and download URL."""
    response = session.get(BTS_URL, timeout=30)
    response.raise_for_status()

    hidden_parser = HiddenFieldParser()
    hidden_parser.feed(response.text)

    required_fields = {
        "__VIEWSTATE",
        "__VIEWSTATEGENERATOR",
        "__EVENTVALIDATION",
    }
    missing_fields = required_fields - hidden_parser.fields.keys()

    if missing_fields:
        raise ValueError(
            f"BTS form is missing required hidden fields: {sorted(missing_fields)}"
        )

    action_parser = FormActionParser()
    action_parser.feed(response.text)

    if not action_parser.download_form_action:
        raise ValueError("BTS download form action was not found")

    download_url = urljoin(BTS_URL, action_parser.download_form_action)

    return hidden_parser.fields, download_url


EARLIEST_YEAR = 1987
LATEST_YEAR = 2026


def build_archive_path(
    raw_root: Path,
    year: int,
    month: int,
) -> Path:
    """Build the destination path for a monthly BTS ZIP archive."""
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    if not EARLIEST_YEAR <= year <= LATEST_YEAR:
        raise ValueError(f"year must be between {EARLIEST_YEAR} and {LATEST_YEAR}")

    return raw_root / str(year) / f"{year}-{month:02d}.zip"


def download_month(
    session: requests.Session,
    year: int,
    month: int,
    raw_root: Path,
) -> Path:
    """Download and validate one BTS monthly archive."""
    destination = build_archive_path(raw_root, year, month)

    if destination.exists():
        raise FileExistsError(
            f"Archive already exists; refusing to overwrite: {destination}"
        )

    hidden_fields, download_url = fetch_download_form(session)
    payload = build_download_payload(hidden_fields, year, month)

    response = session.post(download_url, data=payload, timeout=120)
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".part",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(response.content)

        if not zipfile.is_zipfile(temporary_path):
            raise ValueError("BTS response is not a valid ZIP archive")

        validate_bts_archive(temporary_path)

        # Create the final path without overwriting an existing file.
        # Both paths are on the same filesystem.
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            raise FileExistsError(
                f"Archive already exists; refusing to overwrite: {destination}"
            ) from None

        return destination

    finally:
        temporary_path.unlink(missing_ok=True)
