import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from text_cleaning import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.mark.parametrize("raw,expected", [
    ("priced at 450k", "450000"),
    ("priced at 450K", "450000"),
    ("$450k", "$450000"),
    ("1.2m home", "1200000"),
    ("$1.2m home", "$1200000"),
    ("2m", "2000000"),
    ("0.75m", "750000"),
    ("asking 525.5k", "525500"),
])
def test_price_normalization(cleaner, raw, expected):
    assert expected in cleaner.normalize_prices(raw)


@pytest.mark.parametrize("raw,expected", [
    ("2,000 sqft", "2000 square feet"),
    ("2000 sq ft", "2000 square feet"),
    ("2000 sq. ft.", "2000 square feet"),
    ("850 sf", "850 square feet"),
    ("0.5 ac", "0.5 acres"),
    ("2 acre", "2 acres"),
    ("3 acres", "3 acres"),
])
def test_measurement_normalization(cleaner, raw, expected):
    assert expected in cleaner.normalize_measurements(raw)


@pytest.mark.parametrize("raw,expected", [
    ("3 br home", "3 bedroom home"),
    ("2 ba condo", "2 bathroom condominium"),
    ("mbr suite", "primary bedroom suite"),
    ("full bsmt", "full basement"),
    ("fin bsmt", "finished basement"),
    ("large lr", "large living room"),
    ("formal dr", "formal dining room"),
    ("fam rm", "family room"),
    ("w/ garage", "with garage"),
    ("w/o hoa", "without homeowners association"),
    ("a/c included", "air conditioning included"),
    ("fp in lr", "fireplace in living room"),
    ("att gar", "attached garage"),
    ("det gar", "detached garage"),
    ("sqft", "square feet"),
])
def test_abbreviation_expansion(cleaner, raw, expected):
    assert cleaner.expand_abbreviations(raw).lower() == expected


@pytest.mark.parametrize("raw,expected", [
    ("smart\u00a0home", "smart home"),
    ("“Updated” kitchen", '"Updated" kitchen'),
    ("seller’s suite", "seller's suite"),
    ("great—location", "great-location"),
    ("zero\u200bwidth", "zerowidth"),
])
def test_unicode_normalization(cleaner, raw, expected):
    assert cleaner.normalize_unicode(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("<b>Updated</b> kitchen", " Updated  kitchen"),
    ("Welcome&nbsp;home", "Welcome home"),
    ("<p>Pool</p>", " Pool "),
    ("<script>alert(1)</script>Home", " Home"),
])
def test_html_removal(cleaner, raw, expected):
    assert cleaner.remove_html(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("  hello   world  ", "hello world"),
    ("hello\nworld", "hello world"),
    ("hello\tworld", "hello world"),
])
def test_whitespace_normalization(cleaner, raw, expected):
    assert cleaner.normalize_whitespace(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Wow!!!", "Wow!"),
    ("Great.....home", "Great.home"),
    ("Pool • Spa", "Pool   Spa"),
])
def test_punctuation_normalization(cleaner, raw, expected):
    assert cleaner.normalize_punctuation(raw) == expected


@pytest.mark.parametrize("raw,expected_parts", [
    ("<b>3 BR</b>, 2 BA, 2,000 sqft, $450k.",
     ["3 bedroom", "2 bathroom", "2000 square feet", "$450000"]),
    ("Beautiful mbr w/ fp and 1.2m price.",
     ["primary bedroom", "with fireplace", "1200000"]),
    ("Updated fin bsmt w/o HOA.",
     ["finished basement", "without homeowners association"]),
    (None, [""]),
    ("", [""]),
])
def test_clean_text_pipeline(cleaner, raw, expected_parts):
    cleaned = cleaner.clean_text(raw)
    for expected in expected_parts:
        assert expected.lower() in cleaned.lower()


def test_does_not_expand_inside_words(cleaner):
    assert cleaner.expand_abbreviations("bright room") == "bright room"


def test_profile_keys(cleaner):
    df = pd.DataFrame({"remarks": ["3 br home", None, "<b>$450k</b> 2000 sqft"]})
    profile = cleaner.profile_column(df, "remarks")
    required = {
        "row_count", "null_rate", "avg_length", "median_length", "max_length",
        "price_mentions", "shorthand_price_mentions", "html_presence",
        "non_ascii_rows", "measurement_mentions", "common_abbreviations",
        "common_terms",
    }
    assert required.issubset(profile.keys())


def test_profile_null_rate(cleaner):
    df = pd.DataFrame({"remarks": ["home", None]})
    assert cleaner.profile_column(df, "remarks")["null_rate"] == 0.5


def test_profile_html_detection(cleaner):
    df = pd.DataFrame({"remarks": ["<p>home</p>", "plain"]})
    assert cleaner.profile_column(df, "remarks")["html_presence"] == 1


def test_profile_abbreviation_detection(cleaner):
    df = pd.DataFrame({"remarks": ["3 br 2 ba", "another br"]})
    found = cleaner.profile_column(df, "remarks")["common_abbreviations"]
    result = {item["abbreviation"]: item["count"] for item in found}
    assert result["br"] == 2
    assert result["ba"] == 1


def test_missing_profile_column_raises(cleaner):
    with pytest.raises(KeyError):
        cleaner.profile_column(pd.DataFrame({"x": ["a"]}), "remarks")
