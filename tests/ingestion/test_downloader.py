import csv
import io
import zipfile
from unittest.mock import Mock, patch

import pytest
import requests

from flightops_ml.ingestion.archive import EXPECTED_COLUMNS
from flightops_ml.ingestion.downloader import (
    REQUIRED_COLUMNS,
    FormActionParser,
    FormControlParser,
    HiddenFieldParser,
    build_archive_path,
    build_download_payload,
    download_month,
    fetch_download_form,
    fetch_form_fields,
)

BTS_URL = (
    "https://www.transtats.bts.gov/"
    "DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ"
)


def test_archive_path_uses_year_and_month(tmp_path):
    path = build_archive_path(tmp_path, 2025, 1)

    assert path == tmp_path / "2025" / "2025-01.zip"


def test_archive_path_rejects_invalid_month(tmp_path):
    with pytest.raises(ValueError, match="month"):
        build_archive_path(tmp_path, 2025, 13)


@pytest.mark.parametrize("invalid_year", [1986, 1800, 2027, 2030])
def test_archive_path_rejects_invalid_year(tmp_path, invalid_year):
    with pytest.raises(ValueError, match="year"):
        build_archive_path(tmp_path, invalid_year, 1)


@pytest.mark.parametrize("valid_year", [1987, 2026])
def test_archive_path_accepts_valid_boundary_years(tmp_path, valid_year):
    path = build_archive_path(tmp_path, valid_year, 1)
    assert path == tmp_path / str(valid_year) / f"{valid_year}-01.zip"


def test_hidden_field_parser_extracts_hidden_fields():
    html = """
    <html>
        <input type="hidden" name="__VIEWSTATE" value="state123">
        <input type="hidden" name="__EVENTVALIDATION" value="event456">
        <input type="text" name="username" value="visible">
    </html>
    """

    parser = HiddenFieldParser()
    parser.feed(html)

    assert parser.fields == {
        "__VIEWSTATE": "state123",
        "__EVENTVALIDATION": "event456",
    }


def test_hidden_field_parser_handles_missing_value():
    html = '<input type="hidden" name="affiliate">'

    parser = HiddenFieldParser()
    parser.feed(html)

    assert parser.fields == {"affiliate": ""}


def test_fetch_form_fields_extracts_required_fields():
    response = Mock()
    response.text = """
    <input type="hidden" name="__VIEWSTATE" value="state">
    <input type="hidden" name="__VIEWSTATEGENERATOR" value="generator">
    <input type="hidden" name="__EVENTVALIDATION" value="validation">
    """

    session = Mock()
    session.get.return_value = response

    fields = fetch_form_fields(session)

    assert fields == {
        "__VIEWSTATE": "state",
        "__VIEWSTATEGENERATOR": "generator",
        "__EVENTVALIDATION": "validation",
    }
    session.get.assert_called_once_with(
        "https://www.transtats.bts.gov/"
        "DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ",
        timeout=30,
    )
    response.raise_for_status.assert_called_once()


def test_fetch_form_fields_rejects_missing_required_fields():
    response = Mock()
    response.text = '<input type="hidden" name="__VIEWSTATE" value="state">'

    session = Mock()
    session.get.return_value = response

    with pytest.raises(ValueError, match="missing required hidden fields"):
        fetch_form_fields(session)


def test_form_control_parser_extracts_dropdowns_and_inputs():
    html = """
    <select name="cboYear">
        <option value="2024">2024</option>
        <option value="2025">2025</option>
    </select>
    <input type="checkbox" name="ORIGIN_AIRPORT_ID" checked="checked">
    <input type="submit" name="btnDownload" value="Download">
    """

    parser = FormControlParser()
    parser.feed(html)

    assert parser.select_options["cboYear"] == ["2024", "2025"]
    assert parser.inputs["ORIGIN_AIRPORT_ID"]["checked"] is True
    assert parser.inputs["btnDownload"]["value"] == "Download"


def test_build_download_payload_selects_requested_month_and_columns():
    hidden_fields = {
        "affiliate": "",
        "__VIEWSTATE": "test-state",
        "__VIEWSTATEGENERATOR": "test-generator",
        "__EVENTVALIDATION": "test-validation",
    }

    payload = build_download_payload(hidden_fields, 2025, 1)

    assert payload["cboGeography"] == "All"
    assert payload["cboYear"] == "2025"
    assert payload["cboPeriod"] == "1"
    assert payload["btnDownload"] == "Download"

    for column in REQUIRED_COLUMNS:
        assert payload[column] == "on"

    assert payload["__VIEWSTATE"] == "test-state"


def test_build_download_payload_rejects_missing_hidden_fields():
    with pytest.raises(ValueError, match="Missing required hidden fields"):
        build_download_payload({}, 2025, 1)


def test_build_download_payload_contains_only_requested_columns_and_form_fields():
    hidden_fields = {
        "affiliate": "",
        "__VIEWSTATE": "test-state",
        "__VIEWSTATEGENERATOR": "test-generator",
        "__EVENTVALIDATION": "test-validation",
    }

    payload = build_download_payload(hidden_fields, 2025, 1)

    form_fields = {
        "affiliate",
        "__VIEWSTATE",
        "__VIEWSTATEGENERATOR",
        "__EVENTVALIDATION",
        "cboGeography",
        "cboYear",
        "cboPeriod",
        "btnDownload",
    }

    assert set(payload) == form_fields | set(REQUIRED_COLUMNS)


def test_form_action_parser_finds_download_form():
    html = """
    <form action="https://search.usa.gov/search">
        <input name="query" type="text">
    </form>
    <form action="./DL_SelectFields.aspx?download=1" method="post">
        <select name="cboYear">
            <option value="2025">2025</option>
        </select>
        <select name="cboPeriod">
            <option value="1">January</option>
        </select>
        <input name="btnDownload" type="submit" value="Download">
    </form>
    <form action="Search.asp">
        <input name="search" type="text">
    </form>
    """

    parser = FormActionParser()
    parser.feed(html)

    assert parser.download_form_action == "./DL_SelectFields.aspx?download=1"


def test_form_action_parser_returns_none_without_download_form():
    parser = FormActionParser()
    parser.feed('<form action="Search.asp"><input name="search"></form>')

    assert parser.download_form_action is None


def test_fetch_download_form_returns_hidden_fields_and_resolved_url():
    response = Mock()
    response.text = """
    <input type="hidden" name="__VIEWSTATE" value="state">
    <input type="hidden" name="__VIEWSTATEGENERATOR" value="generator">
    <input type="hidden" name="__EVENTVALIDATION" value="validation">
    <form action="./DL_SelectFields.aspx?download=1">
        <select name="cboYear"><option value="2025">2025</option></select>
        <select name="cboPeriod"><option value="1">January</option></select>
        <input name="btnDownload" type="submit" value="Download">
    </form>
    """

    session = Mock()
    session.get.return_value = response

    fields, url = fetch_download_form(session)

    assert fields["__VIEWSTATE"] == "state"
    assert fields["__EVENTVALIDATION"] == "validation"
    assert url == ("https://www.transtats.bts.gov/DL_SelectFields.aspx?download=1")
    session.get.assert_called_once_with(BTS_URL, timeout=30)


def test_fetch_download_form_rejects_missing_download_form():
    response = Mock()
    response.text = """
    <input type="hidden" name="__VIEWSTATE" value="state">
    <input type="hidden" name="__VIEWSTATEGENERATOR" value="generator">
    <input type="hidden" name="__EVENTVALIDATION" value="validation">
    <form action="Search.asp"><input name="search"></form>
    """

    session = Mock()
    session.get.return_value = response

    with pytest.raises(ValueError, match="download form action"):
        fetch_download_form(session)


def create_valid_bts_zip() -> bytes:
    """Create a small valid BTS-like ZIP archive for testing."""
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(sorted(EXPECTED_COLUMNS))
    writer.writerow(
        ["2025", "1", "2025-01-01", "AA", "JFK", "LAX", "800", "0", "0", "0"]
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.csv", csv_buffer.getvalue())

    return zip_buffer.getvalue()


@patch("flightops_ml.ingestion.downloader.fetch_download_form")
def test_download_month_saves_valid_archive(mock_fetch_form, tmp_path):
    mock_fetch_form.return_value = (
        {
            "__VIEWSTATE": "state",
            "__VIEWSTATEGENERATOR": "generator",
            "__EVENTVALIDATION": "validation",
        },
        "https://example.com/download",
    )

    response = Mock()
    response.content = create_valid_bts_zip()
    response.raise_for_status.return_value = None

    session = Mock()
    session.post.return_value = response

    result = download_month(session, 2025, 1, tmp_path)

    assert result == tmp_path / "2025" / "2025-01.zip"
    assert result.is_file()
    assert zipfile.is_zipfile(result)
    mock_fetch_form.assert_called_once_with(session)
    session.post.assert_called_once()


@patch("flightops_ml.ingestion.downloader.fetch_download_form")
def test_download_month_rejects_html_response(mock_fetch_form, tmp_path):
    mock_fetch_form.return_value = (
        {
            "__VIEWSTATE": "state",
            "__VIEWSTATEGENERATOR": "generator",
            "__EVENTVALIDATION": "validation",
        },
        "https://example.com/download",
    )

    response = Mock()
    response.content = b"<html><body>Server error</body></html>"
    response.raise_for_status.return_value = None

    session = Mock()
    session.post.return_value = response

    with pytest.raises(ValueError, match="not a valid ZIP"):
        download_month(session, 2025, 1, tmp_path)

    assert not (tmp_path / "2025" / "2025-01.zip").exists()
    assert list((tmp_path / "2025").iterdir()) == []


@patch("flightops_ml.ingestion.downloader.fetch_download_form")
def test_download_month_rejects_missing_csv_columns(mock_fetch_form, tmp_path):
    mock_fetch_form.return_value = (
        {
            "__VIEWSTATE": "state",
            "__VIEWSTATEGENERATOR": "generator",
            "__EVENTVALIDATION": "validation",
        },
        "https://example.com/download",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.csv", "YEAR,MONTH\n2025,1\n")

    response = Mock()
    response.content = zip_buffer.getvalue()
    response.raise_for_status.return_value = None

    session = Mock()
    session.post.return_value = response

    with pytest.raises(ValueError, match="Missing expected columns"):
        download_month(session, 2025, 1, tmp_path)

    destination = tmp_path / "2025" / "2025-01.zip"
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


@patch("flightops_ml.ingestion.downloader.fetch_download_form")
def test_download_month_does_not_overwrite_existing_archive(mock_fetch_form, tmp_path):
    mock_fetch_form.return_value = (
        {
            "__VIEWSTATE": "state",
            "__VIEWSTATEGENERATOR": "generator",
            "__EVENTVALIDATION": "validation",
        },
        "https://example.com/download",
    )

    destination = tmp_path / "2025" / "2025-01.zip"
    destination.parent.mkdir(parents=True)
    original_content = b"existing archive must remain untouched"
    destination.write_bytes(original_content)

    response = Mock()
    response.content = create_valid_bts_zip()
    response.raise_for_status.return_value = None

    session = Mock()
    session.post.return_value = response

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        download_month(session, 2025, 1, tmp_path)

    assert destination.read_bytes() == original_content
    assert list(destination.parent.iterdir()) == [destination]


@patch("flightops_ml.ingestion.downloader.fetch_download_form")
def test_download_month_handles_http_failure(mock_fetch_form, tmp_path):
    mock_fetch_form.return_value = (
        {
            "__VIEWSTATE": "state",
            "__VIEWSTATEGENERATOR": "generator",
            "__EVENTVALIDATION": "validation",
        },
        "https://example.com/download",
    )

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    session = Mock()
    session.post.return_value = response

    with pytest.raises(requests.HTTPError, match="500 Server Error"):
        download_month(session, 2025, 1, tmp_path)

    destination = tmp_path / "2025" / "2025-01.zip"
    assert not destination.exists()
    assert not destination.parent.exists()


def test_form_action_parser_handles_nested_form_and_escaped_action():
    html = """
    <form action="./DL_SelectFields.aspx?a=1&amp;b=2">
        <form action="Search.asp">
            <input name="search" type="text">
        </form>
        <select name="cboYear"><option value="2025">2025</option></select>
        <select name="cboPeriod"><option value="1">January</option></select>
        <input name="btnDownload" type="submit" value="Download">
    </form>
    """

    parser = FormActionParser()
    parser.feed(html)

    assert parser.download_form_action == "./DL_SelectFields.aspx?a=1&b=2"
