from html.parser import HTMLParser
from urllib.request import Request, urlopen

REQUIRED_COLUMNS = [
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
]

URL = (
    "https://www.transtats.bts.gov/"
    "DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ"
)

request = Request(URL, headers={"User-Agent": "FlightOpsML/0.1"})

with urlopen(request, timeout=30) as response:
    html = response.read().decode("utf-8", errors="replace")

for term in (
    "cboGeography",
    "chkAllVars",
    "YEAR",
    "FL_DATE",
    "DEP_DEL15",
    "ORIGIN_AIRPORT_ID",
    "btnDownload",
):
    print(f"\n{'=' * 15} {term} {'=' * 15}")

    position = html.find(term)
    if position < 0:
        print("Not found")
        continue

    print(html[max(0, position - 180) : position + 500])


class InputNameParser(HTMLParser):
    """Collect input names from the BTS form."""

    def __init__(self):
        super().__init__()
        self.input_names = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return

        attributes = dict(attrs)
        name = attributes.get("name")

        if name:
            self.input_names.add(name)


class HiddenFieldParser(HTMLParser):
    """Collect names of hidden form fields."""

    def __init__(self):
        super().__init__()
        self.hidden_names = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return

        attributes = dict(attrs)

        if attributes.get("type", "").lower() == "hidden":
            name = attributes.get("name")
            if name:
                self.hidden_names.append(name)


input_parser = InputNameParser()
input_parser.feed(html)

missing_columns = set(REQUIRED_COLUMNS) - input_parser.input_names

print("\nRequired column verification:")
print(f"Found: {len(REQUIRED_COLUMNS) - len(missing_columns)}/{len(REQUIRED_COLUMNS)}")

if missing_columns:
    print("Missing:", sorted(missing_columns))
else:
    print("All required columns exist.")

hidden_parser = HiddenFieldParser()
hidden_parser.feed(html)

print("\nHidden form field names:")
for name in hidden_parser.hidden_names:
    print(f"- {name}")


class FormDetailsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current_form = None
        self.checkbox_names = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)

        if tag.lower() == "form":
            self.current_form = {
                "action": attributes.get("action"),
                "method": attributes.get("method", "get").lower(),
            }
            self.forms.append(self.current_form)

        elif tag.lower() == "input":
            if attributes.get("type", "").lower() == "checkbox":
                name = attributes.get("name")
                if name:
                    self.checkbox_names.add(name)

    def handle_endtag(self, tag):
        if tag.lower() == "form":
            self.current_form = None


details = FormDetailsParser()
details.feed(html)

print("\nForm details:")
for index, form in enumerate(details.forms):
    print(f"Form {index}: method={form['method']!r}, action={form['action']!r}")

print("\nRequested columns missing from checkbox inputs:")
print(sorted(set(REQUIRED_COLUMNS) - details.checkbox_names))
